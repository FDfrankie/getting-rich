import numpy as np

from crypto_trading.math_models import (
    estimate_hurst_exponent,
    estimate_ou_parameters,
    fractional_kelly_size,
)


def test_ou_estimator_on_mean_reverting_series():
    rng = np.random.default_rng(42)
    n = 300
    x = np.zeros(n)
    theta_true = 0.15
    for t in range(1, n):
        x[t] = x[t - 1] + theta_true * (0.0 - x[t - 1]) + rng.normal(0, 0.01)
    prices = 100.0 * np.exp(x)
    ou = estimate_ou_parameters(prices)
    assert ou.is_mean_reverting is True
    assert 1.5 <= ou.half_life_bars <= 20.0


def test_hurst_exponent_distinguishes_trend_and_mean_reversion():
    rng = np.random.default_rng(99)
    n = 400
    # Persistent trending series (positively autocorrelated returns)
    trend_rets = np.zeros(n)
    for t_i in range(1, n):
        trend_rets[t_i] = 0.65 * trend_rets[t_i - 1] + rng.normal(0, 0.001)
    trend_prices = 100.0 * np.exp(np.cumsum(trend_rets))
    h_trend = estimate_hurst_exponent(trend_prices)

    # Mean-reverting series
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + rng.normal(0, 0.01)
    mr_prices = 100.0 * np.exp(x)
    h_mr = estimate_hurst_exponent(mr_prices)

    assert h_trend > 0.55
    assert h_mr < 0.45


def test_fractional_kelly_respects_cost_deadband_and_cap():
    assert fractional_kelly_size(0.0002, 0.01, roundtrip_cost=0.0004) == 0.0
    pos = fractional_kelly_size(0.002, 0.01, roundtrip_cost=0.0004, kelly_fraction=0.25, max_leverage=1.0)
    assert 0.0 < pos <= 1.0
    neg = fractional_kelly_size(-0.05, 0.01, roundtrip_cost=0.0004, kelly_fraction=0.5, max_leverage=1.0)
    assert neg == -1.0
