"""Offline unit tests for LocalCrossSectionalScreener parallel PV pre-screening and non-PV pass-through."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from worldquant_brain.local_cs_screener import (
    LocalCrossSectionalScreener,
    evaluate_fastexpr_panel,
    is_price_volume_only,
)


def _make_synthetic_us_panel(n_days: int = 180, n_stocks: int = 20) -> dict[str, pd.DataFrame]:
    rng = np.random.RandomState(42)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="B")
    cols = [f"STK_{i:02d}" for i in range(n_stocks)]
    # Create mild mean-reverting cross-sectional returns so short-term reversal has positive local Sharpe
    noise = rng.normal(0.0, 0.018, size=(n_days, n_stocks))
    rets = np.zeros_like(noise)
    for t in range(1, n_days):
        rets[t] = -0.25 * rets[t - 1] + noise[t]
    close = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=cols)
    open_ = close * (1.0 + rng.normal(0.0, 0.003, size=(n_days, n_stocks)))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.01, size=(n_days, n_stocks)))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.01, size=(n_days, n_stocks)))
    volume = pd.DataFrame(rng.uniform(1e5, 5e6, size=(n_days, n_stocks)), index=dates, columns=cols)
    vwap = (high + low + close) / 3.0
    return {
        "close": close,
        "open": open_,
        "high": high,
        "low": low,
        "volume": volume,
        "vwap": vwap,
        "returns": close.pct_change().fillna(0.0),
    }


class TestLocalCrossSectionalScreener(unittest.TestCase):
    def test_pv_detection(self) -> None:
        self.assertTrue(is_price_volume_only("hump(group_neutralize(winsorize(rank(-1 * returns), std=4), subindustry), hump=0.01)"))
        self.assertFalse(is_price_volume_only("rank(operating_cashflow / (cap + debt + 0.001))"))

    def test_parallel_screening_filters_negative_and_dedups_correlated(self) -> None:
        panel = _make_synthetic_us_panel()
        screener = LocalCrossSectionalScreener(
            panel=panel,
            min_local_sharpe=0.50,
            max_local_turnover=1.95,
            max_peer_corr=0.65,
            max_workers=4,
        )
        candidates = [
            {"name": "Good_Rev", "expression": "group_neutralize(rank(-1 * returns), subindustry)", "domain": "PV_RESIDUAL_HUMP"},
            {"name": "Clone_Rev", "expression": "group_neutralize(rank(-2 * returns), subindustry)", "domain": "PV_RESIDUAL_HUMP"},
            {"name": "Bad_Mom", "expression": "group_neutralize(rank(returns), subindustry)", "domain": "PV_RESIDUAL_HUMP"},
            {"name": "Fund_Pass", "expression": "rank(ebitda / (cap + debt + 0.001))", "domain": "FUNDAMENTAL_CF_EV"},
        ]
        selected = screener.screen_batch(candidates, top_k=5)
        names = [c["name"] for c in selected]
        self.assertIn("Good_Rev", names)
        self.assertNotIn("Clone_Rev", names)  # filtered out by pairwise peer corr >= 0.65
        self.assertNotIn("Bad_Mom", names)    # filtered out by negative local Sharpe
        self.assertIn("Fund_Pass", names)     # non-PV passes straight through

    def test_evaluate_fastexpr_panel_runs_complex_ops(self) -> None:
        panel = _make_synthetic_us_panel(n_days=80, n_stocks=10)
        df = evaluate_fastexpr_panel(
            "hump(group_neutralize(winsorize(rank(-1 * ts_regression(returns, ts_mean(returns, 5), 20, rettype=0)), std=4), subindustry), hump=0.01)",
            panel,
        )
        self.assertEqual(df.shape, panel["close"].shape)
        self.assertTrue(df.notna().sum().sum() > 0)


if __name__ == "__main__":
    unittest.main()
