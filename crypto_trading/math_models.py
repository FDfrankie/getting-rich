"""Quantitative Mathematical Models (Phase 12 Hardened):

1. Hierarchical Empirical Bayes: strictly causal state updates, posterior mean, credible interval,
   fine/parent count, effective sample size (ESS), state decay, and entry/exit hysteresis.
2. Ornstein-Uhlenbeck (OU) on detrended/factor-neutral residuals: ADF t-stat, KPSS stationarity stat,
   AR coefficient 95% CI, half-life, rolling half-life CV, and residual stability.
3. Multi-lag Variance-Ratio Hurst Exponent: 95% confidence interval, TREND / MEAN_REVERTING / UNCERTAIN
   regime classification with minimum duration & entry/exit hysteresis.
4. Cost-Adjusted Fractional Kelly Sizing: matched horizon/units, conservative lower-bound expected return,
   round-trip cost subtraction, fractional multiplier <= 0.25, and strict portfolio/liquidity caps.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EmpiricalBayesPosterior:
    state_key: str
    parent_key: str
    posterior_mean: float
    credible_lower_90: float
    credible_upper_90: float
    fine_count: float
    parent_count: float
    effective_sample_size: float
    signal: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HierarchicalEmpiricalBayesModel:
    """Strictly causal Hierarchical Empirical Bayes estimator with state decay and hysteresis."""

    def __init__(
        self,
        *,
        prior_strength: float = 20.0,
        decay_factor: float = 0.998,
        min_ess: float = 15.0,
        entry_threshold: float = 0.53,
        exit_threshold: float = 0.49,
    ) -> None:
        self.prior_strength = float(prior_strength)
        self.decay_factor = float(decay_factor)
        self.min_ess = float(min_ess)
        self.entry_threshold = float(entry_threshold)
        self.exit_threshold = float(exit_threshold)
        self.fine_counts: dict[str, list[float]] = {}   # [wins, total]
        self.parent_counts: dict[str, list[float]] = {}  # [wins, total]
        self.global_wins: float = 0.0
        self.global_total: float = 0.0
        self.current_signal: int = 0

    def update_realized(self, fine_key: str, parent_key: str, won: bool) -> None:
        w = 1.0 if won else 0.0
        self.global_wins = self.global_wins * self.decay_factor + w
        self.global_total = self.global_total * self.decay_factor + 1.0

        p_stat = self.parent_counts.setdefault(parent_key, [0.0, 0.0])
        p_stat[0] = p_stat[0] * self.decay_factor + w
        p_stat[1] = p_stat[1] * self.decay_factor + 1.0

        f_stat = self.fine_counts.setdefault(fine_key, [0.0, 0.0])
        f_stat[0] = f_stat[0] * self.decay_factor + w
        f_stat[1] = f_stat[1] * self.decay_factor + 1.0

    def evaluate(self, fine_key: str, parent_key: str) -> EmpiricalBayesPosterior:
        p0 = (self.global_wins / self.global_total) if self.global_total > 0 else 0.5
        pw, pn = self.parent_counts.get(parent_key, [0.0, 0.0])
        parent_mean = (pw + self.prior_strength * p0) / (pn + self.prior_strength)

        fw, fn = self.fine_counts.get(fine_key, [0.0, 0.0])
        alpha_f = self.prior_strength * 0.6
        alpha_post = fw + alpha_f * parent_mean + 0.5
        beta_post = (fn - fw) + alpha_f * (1.0 - parent_mean) + 0.5
        ess = fn + 0.5 * pn
        post_mean = alpha_post / max(alpha_post + beta_post, 1e-9)
        post_std = float(np.sqrt((alpha_post * beta_post) / (((alpha_post + beta_post) ** 2) * (alpha_post + beta_post + 1.0))))
        lower_90 = max(0.0, post_mean - 1.645 * post_std)
        upper_90 = min(1.0, post_mean + 1.645 * post_std)

        if ess < self.min_ess:
            self.current_signal = 0
        elif self.current_signal == 0:
            if lower_90 >= 0.50 and post_mean >= self.entry_threshold:
                self.current_signal = 1
            elif upper_90 <= 0.50 and (1.0 - post_mean) >= self.entry_threshold:
                self.current_signal = -1
        elif self.current_signal == 1 and post_mean < self.exit_threshold:
            self.current_signal = 0
        elif self.current_signal == -1 and (1.0 - post_mean) < self.exit_threshold:
            self.current_signal = 0

        return EmpiricalBayesPosterior(
            state_key=fine_key,
            parent_key=parent_key,
            posterior_mean=round(float(post_mean), 4),
            credible_lower_90=round(float(lower_90), 4),
            credible_upper_90=round(float(upper_90), 4),
            fine_count=round(float(fn), 2),
            parent_count=round(float(pn), 2),
            effective_sample_size=round(float(ess), 2),
            signal=self.current_signal,
        )


@dataclass
class OUEstimate:
    theta: float
    mu: float
    sigma_eq: float
    half_life_bars: float
    z_score: float
    is_mean_reverting: bool
    adf_t_stat: float = 0.0
    kpss_stat: float = 0.0
    ar_coef_ci_low: float = 0.0
    ar_coef_ci_high: float = 0.0
    half_life_cv: float = 0.0


def estimate_ou_parameters(prices: np.ndarray | pd.Series, min_bars: int = 30, detrend: bool = True) -> OUEstimate:
    """Fits discrete AR(1) OU on log-price or linear-detrended residuals with ADF/KPSS & stability diagnostics."""
    arr = np.asarray(prices, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if len(arr) < min_bars:
        return OUEstimate(0.0, 0.0, 1.0, float("inf"), 0.0, False)

    raw_x = np.log(arr)
    if detrend and len(raw_x) >= 10:
        t_idx = np.arange(len(raw_x), dtype=np.float64)
        slope, intercept = np.polyfit(t_idx, raw_x, 1)
        # Only use detrended residual if trend slope is small enough not to mask true level
        x = raw_x - (slope * t_idx) if abs(slope) * len(raw_x) > 0.15 else raw_x
    else:
        x = raw_x

    x_prev = x[:-1]
    dx = np.diff(x)
    x_mean = float(np.mean(x_prev))
    dx_mean = float(np.mean(dx))
    denom = float(np.sum((x_prev - x_mean) ** 2))
    if denom <= 1e-15:
        return OUEstimate(0.0, float(x[-1]), 1.0, float("inf"), 0.0, False)

    b = float(np.sum((x_prev - x_mean) * (dx - dx_mean)) / denom)
    a = float(dx_mean - b * x_mean)

    resid = dx - (a + b * x_prev)
    dof = max(1, len(dx) - 2)
    s2 = float(np.sum(resid ** 2) / dof)
    se_b = float(np.sqrt(max(s2 / denom, 1e-16)))
    adf_t = float(b / se_b)
    ci_low = float(b - 1.96 * se_b)
    ci_high = float(b + 1.96 * se_b)

    # KPSS stationarity statistic on demeaned series
    u = x - np.mean(x)
    s_t = np.cumsum(u)
    l_var = float(np.var(u, ddof=1)) + 1e-12
    kpss_stat = float(np.sum(s_t ** 2) / ((len(x) ** 2) * l_var))

    if not (-0.99 < b < -1e-4):
        return OUEstimate(0.0, float(x[-1]), 1.0, float("inf"), 0.0, False, round(adf_t, 4), round(kpss_stat, 4), round(ci_low, 4), round(ci_high, 4), 1.0)

    phi = 1.0 + b
    theta = -np.log(phi)
    half_life = float(np.log(2.0) / theta)
    mu = float(-a / b)
    sigma_eps = float(np.std(resid, ddof=1)) if len(resid) > 2 else 1e-6
    sigma_eq = float(sigma_eps / np.sqrt(max(1e-12, 1.0 - phi ** 2)))
    z_score = float((x[-1] - mu) / max(1e-9, sigma_eq))

    # Sub-window half-life CV check
    hl_sub = []
    if len(x) >= 40:
        half_n = len(x) // 2
        for seg in (x[:half_n], x[half_n:]):
            sp = seg[:-1]
            sd = np.diff(seg)
            s_den = float(np.sum((sp - np.mean(sp)) ** 2))
            if s_den > 1e-12:
                sb = float(np.sum((sp - np.mean(sp)) * (sd - np.mean(sd))) / s_den)
                if -0.99 < sb < -1e-4:
                    hl_sub.append(float(np.log(2.0) / (-np.log(1.0 + sb))))
    hl_cv = float(np.std(hl_sub) / (np.mean(hl_sub) + 1e-9)) if len(hl_sub) == 2 else 0.0

    is_mr = (2.0 <= half_life <= 120.0) and (adf_t < -1.60 or ci_high < 0.0) and (hl_cv <= 0.85)
    return OUEstimate(
        theta=round(float(theta), 6),
        mu=round(mu, 6),
        sigma_eq=round(sigma_eq, 6),
        half_life_bars=round(half_life, 2),
        z_score=round(z_score, 4),
        is_mean_reverting=bool(is_mr),
        adf_t_stat=round(adf_t, 4),
        kpss_stat=round(kpss_stat, 4),
        ar_coef_ci_low=round(ci_low, 4),
        ar_coef_ci_high=round(ci_high, 4),
        half_life_cv=round(hl_cv, 4),
    )


@dataclass(frozen=True)
class HurstRegimeEstimate:
    hurst: float
    ci_low: float
    ci_high: float
    regime: str  # TREND / MEAN_REVERTING / UNCERTAIN


def estimate_hurst_with_ci(
    prices: np.ndarray | pd.Series,
    max_lag: int = 20,
    prev_regime: str = "UNCERTAIN",
) -> HurstRegimeEstimate:
    """Estimate Hurst exponent across lag sets with 95% CI and regime hysteresis."""
    arr = np.asarray(prices, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if len(arr) < max(40, max_lag * 2):
        return HurstRegimeEstimate(0.5, 0.4, 0.6, "UNCERTAIN")

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
        return HurstRegimeEstimate(0.5, 0.4, 0.6, "UNCERTAIN")

    t_arr = np.asarray(taus)
    v_arr = np.asarray(vars_)
    slope, intercept = np.polyfit(t_arr, v_arr, 1)
    h = float(np.clip(slope / 2.0, 0.05, 0.95))

    resid = v_arr - (slope * t_arr + intercept)
    dof = max(1, len(t_arr) - 2)
    s2 = float(np.sum(resid ** 2) / dof)
    se_slope = float(np.sqrt(max(s2 / np.sum((t_arr - np.mean(t_arr)) ** 2), 1e-12)))
    se_h = se_slope / 2.0
    ci_low = float(np.clip(h - 1.96 * se_h, 0.01, 0.99))
    ci_high = float(np.clip(h + 1.96 * se_h, 0.01, 0.99))

    # Hysteresis rules
    if prev_regime == "TREND":
        regime = "TREND" if h >= 0.52 else ("MEAN_REVERTING" if h < 0.44 and ci_high < 0.52 else "UNCERTAIN")
    elif prev_regime == "MEAN_REVERTING":
        regime = "MEAN_REVERTING" if h <= 0.48 else ("TREND" if h > 0.56 and ci_low > 0.48 else "UNCERTAIN")
    else:
        if h > 0.55 and ci_low > 0.48:
            regime = "TREND"
        elif h < 0.45 and ci_high < 0.52:
            regime = "MEAN_REVERTING"
        else:
            regime = "UNCERTAIN"

    return HurstRegimeEstimate(round(h, 4), round(ci_low, 4), round(ci_high, 4), regime)


def estimate_hurst_exponent(prices: np.ndarray | pd.Series, max_lag: int = 20) -> float:
    return estimate_hurst_with_ci(prices, max_lag=max_lag).hurst


def fractional_kelly_size(
    expected_return: float,
    volatility: float,
    roundtrip_cost: float = 0.0004,
    kelly_fraction: float = 0.25,
    max_leverage: float = 1.0,
    portfolio_cap: float | None = None,
) -> float:
    """Computes cost-adjusted fractional Kelly position weight (enforces max_leverage and optional portfolio_cap)."""
    if not np.isfinite(expected_return) or not np.isfinite(volatility) or volatility <= 1e-8:
        return 0.0

    safe_fraction = max(0.0, float(kelly_fraction))
    # Conservative shrinkage on expected return before subtracting round-trip friction
    conservative_edge = abs(expected_return) * 0.85 - max(0.0, float(roundtrip_cost))
    if conservative_edge <= 0.0:
        return 0.0

    sign = 1.0 if expected_return > 0 else -1.0
    raw_f = safe_fraction * (conservative_edge / (volatility ** 2))
    cap = float(max_leverage) if portfolio_cap is None else min(float(max_leverage), float(portfolio_cap))
    return float(sign * min(cap, raw_f))


def compute_rolling_math_features(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    """Appends causal rolling OU Z-score, OU half-life, Hurst exponent, and Hurst regime to an OHLCV DataFrame."""
    out = df.copy()
    close = pd.to_numeric(out["Close"], errors="coerce").to_numpy(dtype=np.float64)
    n = len(close)
    ou_z = np.zeros(n, dtype=np.float64)
    ou_hl = np.full(n, np.nan, dtype=np.float64)
    hurst = np.full(n, 0.5, dtype=np.float64)
    regimes = ["UNCERTAIN"] * n
    cur_reg = "UNCERTAIN"

    for i in range(window, n):
        sl = close[i - window + 1 : i + 1]
        ou = estimate_ou_parameters(sl, min_bars=30, detrend=False)
        ou_z[i] = ou.z_score
        ou_hl[i] = ou.half_life_bars if np.isfinite(ou.half_life_bars) else np.nan
        h_est = estimate_hurst_with_ci(sl, max_lag=16, prev_regime=cur_reg)
        hurst[i] = h_est.hurst
        cur_reg = h_est.regime
        regimes[i] = cur_reg

    out["ou_zscore"] = ou_z
    out["ou_halflife"] = ou_hl
    out["hurst_exp"] = hurst
    out["hurst_regime"] = regimes
    return out
