"""
Quantitative Mathematical Models for Multi-Timeframe Crypto Alpha:
  1. Ornstein-Uhlenbeck (OU) Mean-Reversion Model: dX_t = theta * (mu - X_t) dt + sigma dW_t
  2. Variance-Ratio Hurst Exponent (H): distinguishes Trending (H > 0.55) vs Mean-Reverting (H < 0.45)
  3. Cost-Adjusted Fractional Kelly Sizing: f* = kappa * (E[r] - cost) / sigma^2
All estimators operate strictly causally on historical slices [:t+1].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OUEstimate:
    theta: float
    mu: float
    sigma_eq: float
    half_life_bars: float
    z_score: float
    is_mean_reverting: bool


def estimate_ou_parameters(prices: np.ndarray | pd.Series, min_bars: int = 30) -> OUEstimate:
    """
    Fits discrete AR(1) representation of Ornstein-Uhlenbeck on log-prices:
      x_t - x_{t-1} = a + b * x_{t-1} + eps_t
    """
    arr = np.asarray(prices, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if len(arr) < min_bars:
        return OUEstimate(0.0, 0.0, 1.0, float("inf"), 0.0, False)

    x = np.log(arr)
    x_prev = x[:-1]
    dx = np.diff(x)

    # OLS: dx = a + b * x_prev
    x_mean = np.mean(x_prev)
    dx_mean = np.mean(dx)
    denom = np.sum((x_prev - x_mean) ** 2)
    if denom <= 1e-15:
        return OUEstimate(0.0, float(x[-1]), 1.0, float("inf"), 0.0, False)

    b = float(np.sum((x_prev - x_mean) * (dx - dx_mean)) / denom)
    a = float(dx_mean - b * x_mean)

    # Require -1 < b < 0 for stationary mean-reversion
    if not (-0.99 < b < -1e-4):
        return OUEstimate(0.0, float(x[-1]), 1.0, float("inf"), 0.0, False)

    phi = 1.0 + b
    theta = -np.log(phi)
    half_life = float(np.log(2.0) / theta)
    mu = float(-a / b)

    resid = dx - (a + b * x_prev)
    sigma_eps = float(np.std(resid, ddof=1)) if len(resid) > 2 else 1e-6
    sigma_eq = float(sigma_eps / np.sqrt(max(1e-12, 1.0 - phi ** 2)))
    z_score = float((x[-1] - mu) / max(1e-9, sigma_eq))

    return OUEstimate(
        theta=round(float(theta), 6),
        mu=round(mu, 6),
        sigma_eq=round(sigma_eq, 6),
        half_life_bars=round(half_life, 2),
        z_score=round(z_score, 4),
        is_mean_reverting=(2.0 <= half_life <= 120.0),
    )


def estimate_hurst_exponent(prices: np.ndarray | pd.Series, max_lag: int = 20) -> float:
    """
    Estimates Hurst Exponent H via log-log scaling of lagged return variance:
      Var(x_{t+tau} - x_t) ~ tau^{2H}
    H > 0.55 => Trending / Persistent
    H < 0.45 => Mean-Reverting / Anti-persistent
    H ~ 0.50 => Random Walk
    """
    arr = np.asarray(prices, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if len(arr) < max(40, max_lag * 2):
        return 0.5

    x = np.log(arr)
    lags = np.arange(2, max_lag + 1)
    taus = []
    vars_ = []
    for lag in lags:
        diff = x[lag:] - x[:-lag]
        v = float(np.var(diff))
        if v > 1e-16:
            taus.append(np.log(lag))
            vars_.append(np.log(v))

    if len(taus) < 4:
        return 0.5

    slope, _ = np.polyfit(taus, vars_, 1)
    h = float(np.clip(slope / 2.0, 0.05, 0.95))
    return round(h, 4)


def fractional_kelly_size(
    expected_return: float,
    volatility: float,
    roundtrip_cost: float = 0.0004,
    kelly_fraction: float = 0.25,
    max_leverage: float = 1.0,
) -> float:
    """
    Computes cost-adjusted fractional Kelly position weight in [-max_leverage, max_leverage].
    Returns 0.0 if expected net edge does not clear transaction costs.
    """
    if not np.isfinite(expected_return) or not np.isfinite(volatility) or volatility <= 1e-8:
        return 0.0

    edge = abs(expected_return) - max(0.0, roundtrip_cost)
    if edge <= 0.0:
        return 0.0

    sign = 1.0 if expected_return > 0 else -1.0
    raw_f = kelly_fraction * (edge / (volatility ** 2))
    return float(sign * min(max_leverage, raw_f))


def compute_rolling_math_features(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    """
    Appends causal rolling OU Z-score, OU half-life, and Hurst exponent to an OHLCV DataFrame.
    """
    out = df.copy()
    close = pd.to_numeric(out["Close"], errors="coerce").to_numpy(dtype=np.float64)
    n = len(close)
    ou_z = np.zeros(n, dtype=np.float64)
    ou_hl = np.full(n, np.nan, dtype=np.float64)
    hurst = np.full(n, 0.5, dtype=np.float64)

    for i in range(window, n):
        sl = close[i - window + 1 : i + 1]
        ou = estimate_ou_parameters(sl, min_bars=30)
        ou_z[i] = ou.z_score
        ou_hl[i] = ou.half_life_bars if np.isfinite(ou.half_life_bars) else np.nan
        hurst[i] = estimate_hurst_exponent(sl, max_lag=16)

    out["ou_zscore"] = ou_z
    out["ou_halflife"] = ou_hl
    out["hurst_exp"] = hurst
    return out
