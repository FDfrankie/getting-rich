"""Causal Binance USDT-M Derivatives Data Loader (Funding Rate, Open Interest, Top Trader L/S Ratio).

All merges strictly use backward `merge_asof` on UTC timestamps (zero lookahead bias).
Gracefully falls back to neutral columns if offline or rate-limited.
"""
from __future__ import annotations

import json
import pathlib
import time
import urllib.parse
import urllib.request
from typing import Any

import numpy as np
import pandas as pd


def _http_get_json(url: str, timeout: int = 15) -> list[dict[str, Any]]:
    req = urllib.request.Request(url, headers={"User-Agent": "AntigravityQuant-P54/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data if isinstance(data, list) else []


def fetch_funding_rate_history(
    symbol: str = "BTCUSDT",
    limit: int = 500,
    cache_dir: pathlib.Path | None = None,
) -> pd.DataFrame:
    """Fetch real Binance USDT-M 8h funding rate history."""
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cf = cache_dir / f"{symbol}_funding_{limit}.csv"
        if cf.exists() and (time.time() - cf.stat().st_mtime) < 3600:
            return pd.read_csv(cf)
    try:
        url = "https://fapi.binance.com/fapi/v1/fundingRate?" + urllib.parse.urlencode(
            {"symbol": symbol.upper(), "limit": min(1000, int(limit))}
        )
        rows = _http_get_json(url)
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["OpenTime", "FUNDING_RATE_8H"])
        df["OpenTime"] = pd.to_numeric(df["fundingTime"], errors="coerce").astype("int64")
        df["FUNDING_RATE_8H"] = pd.to_numeric(df["fundingRate"], errors="coerce").fillna(0.0)
        df = df[["OpenTime", "FUNDING_RATE_8H"]].drop_duplicates("OpenTime").sort_values("OpenTime").reset_index(drop=True)
        if cache_dir is not None:
            df.to_csv(cache_dir / f"{symbol}_funding_{limit}.csv", index=False)
        return df
    except Exception:
        return pd.DataFrame(columns=["OpenTime", "FUNDING_RATE_8H"])


def fetch_open_interest_history(
    symbol: str = "BTCUSDT",
    period: str = "5m",
    limit: int = 500,
    cache_dir: pathlib.Path | None = None,
) -> pd.DataFrame:
    """Fetch real Binance USDT-M Open Interest history (period in 5m, 15m, 30m, 1h)."""
    valid_period = period if period in {"5m", "15m", "30m", "1h", "2h", "4h"} else "5m"
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cf = cache_dir / f"{symbol}_oi_{valid_period}_{limit}.csv"
        if cf.exists() and (time.time() - cf.stat().st_mtime) < 1800:
            return pd.read_csv(cf)
    try:
        url = "https://fapi.binance.com/futures/data/openInterestHist?" + urllib.parse.urlencode(
            {"symbol": symbol.upper(), "period": valid_period, "limit": min(500, int(limit))}
        )
        rows = _http_get_json(url)
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["OpenTime", "OPEN_INTEREST"])
        df["OpenTime"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("int64")
        df["OPEN_INTEREST"] = pd.to_numeric(df["sumOpenInterestValue"], errors="coerce").fillna(0.0)
        df = df[["OpenTime", "OPEN_INTEREST"]].drop_duplicates("OpenTime").sort_values("OpenTime").reset_index(drop=True)
        if cache_dir is not None:
            df.to_csv(cache_dir / f"{symbol}_oi_{valid_period}_{limit}.csv", index=False)
        return df
    except Exception:
        return pd.DataFrame(columns=["OpenTime", "OPEN_INTEREST"])


def fetch_top_trader_ls_ratio(
    symbol: str = "BTCUSDT",
    period: str = "15m",
    limit: int = 500,
    cache_dir: pathlib.Path | None = None,
) -> pd.DataFrame:
    """Fetch Binance Top Trader Long/Short Position Ratio."""
    valid_period = period if period in {"5m", "15m", "30m", "1h", "2h", "4h"} else "15m"
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cf = cache_dir / f"{symbol}_topls_{valid_period}_{limit}.csv"
        if cf.exists() and (time.time() - cf.stat().st_mtime) < 1800:
            return pd.read_csv(cf)
    try:
        url = "https://fapi.binance.com/futures/data/topLongShortPositionRatio?" + urllib.parse.urlencode(
            {"symbol": symbol.upper(), "period": valid_period, "limit": min(500, int(limit))}
        )
        rows = _http_get_json(url)
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["OpenTime", "TOP_LS_RATIO"])
        df["OpenTime"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("int64")
        df["TOP_LS_RATIO"] = pd.to_numeric(df["longShortRatio"], errors="coerce").fillna(1.0)
        df = df[["OpenTime", "TOP_LS_RATIO"]].drop_duplicates("OpenTime").sort_values("OpenTime").reset_index(drop=True)
        if cache_dir is not None:
            df.to_csv(cache_dir / f"{symbol}_topls_{valid_period}_{limit}.csv", index=False)
        return df
    except Exception:
        return pd.DataFrame(columns=["OpenTime", "TOP_LS_RATIO"])


def classify_oi_price_quadrant(price_ret: pd.Series, oi_change_pct: pd.Series) -> pd.Series:
    """Classify institutional OI-Price quadrant (strictly causal):
      - FRESH_LONG:  Price > 0 & OI > 0 (new long leverage entering -> strong trend)
      - SHORT_COVER: Price > 0 & OI < 0 (shorts closing -> weaker bounce)
      - FRESH_SHORT: Price < 0 & OI > 0 (new short leverage entering -> strong downtrend)
      - LIQ_FLUSH:   Price < 0 & OI < 0 (long liquidations -> exhaustion reversal zone)
    """
    p_up = price_ret > 0.0005
    p_dn = price_ret < -0.0005
    oi_up = oi_change_pct > 0.0005
    oi_dn = oi_change_pct < -0.0005
    q = np.where(
        p_up & oi_up,
        "FRESH_LONG",
        np.where(
            p_up & oi_dn,
            "SHORT_COVER",
            np.where(p_dn & oi_up, "FRESH_SHORT", np.where(p_dn & oi_dn, "LIQ_FLUSH", "NEUTRAL")),
        ),
    )
    return pd.Series(q, index=price_ret.index)


def enrich_ohlcv_with_derivatives(
    df: pd.DataFrame,
    funding_df: pd.DataFrame | None = None,
    oi_df: pd.DataFrame | None = None,
    ls_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Causally attach Funding Rate, Open Interest, OI-Price Quadrant & Top Trader L/S Z-score."""
    out = df.copy()
    if "OpenTime" not in out.columns:
        out["FUNDING_RATE_8H"] = 0.0001
        out["OI_CHANGE_PCT"] = 0.0
        out["OI_QUADRANT"] = "NEUTRAL"
        out["TOP_LS_ZSCORE"] = 0.0
        return out

    out["OpenTime"] = pd.to_numeric(out["OpenTime"], errors="coerce").astype("int64")
    out = out.sort_values("OpenTime").reset_index(drop=True)

    if funding_df is not None and not funding_df.empty:
        f_sorted = funding_df[["OpenTime", "FUNDING_RATE_8H"]].copy()
        f_sorted["OpenTime"] = pd.to_numeric(f_sorted["OpenTime"], errors="coerce").astype("int64")
        f_sorted = f_sorted.sort_values("OpenTime")
        out = pd.merge_asof(out, f_sorted, on="OpenTime", direction="backward")
        out["FUNDING_RATE_8H"] = pd.to_numeric(out["FUNDING_RATE_8H"], errors="coerce").fillna(0.0001)
    else:
        out["FUNDING_RATE_8H"] = 0.0001

    if oi_df is not None and not oi_df.empty:
        o_sorted = oi_df[["OpenTime", "OPEN_INTEREST"]].copy()
        o_sorted["OpenTime"] = pd.to_numeric(o_sorted["OpenTime"], errors="coerce").astype("int64")
        o_sorted = o_sorted.sort_values("OpenTime")
        out = pd.merge_asof(out, o_sorted, on="OpenTime", direction="backward")
        oi_s = pd.to_numeric(out["OPEN_INTEREST"], errors="coerce").ffill().fillna(0.0)
        out["OI_CHANGE_PCT"] = oi_s.pct_change(5).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    else:
        out["OI_CHANGE_PCT"] = 0.0

    close_s = pd.to_numeric(out["Close"], errors="coerce")
    out["OI_QUADRANT"] = classify_oi_price_quadrant(close_s.pct_change(5).fillna(0.0), out["OI_CHANGE_PCT"])

    if ls_df is not None and not ls_df.empty:
        l_sorted = ls_df[["OpenTime", "TOP_LS_RATIO"]].copy()
        l_sorted["OpenTime"] = pd.to_numeric(l_sorted["OpenTime"], errors="coerce").astype("int64")
        l_sorted = l_sorted.sort_values("OpenTime")
        out = pd.merge_asof(out, l_sorted, on="OpenTime", direction="backward")
        ls_s = pd.to_numeric(out["TOP_LS_RATIO"], errors="coerce").ffill().fillna(1.0)
        mu = ls_s.rolling(60, min_periods=10).mean()
        sd = ls_s.rolling(60, min_periods=10).std().replace(0, np.nan)
        out["TOP_LS_ZSCORE"] = ((ls_s - mu) / sd).fillna(0.0).clip(-4.0, 4.0)
    else:
        out["TOP_LS_ZSCORE"] = 0.0

    return out
