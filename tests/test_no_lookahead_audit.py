import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crypto_trading.feature_engineer import CryptoFeatureEngineer
from taiwan_stock.ml_ranker import TaiwanStockMLRanker


def bars(n=180, freq="1h", seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq=freq)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame(
        {
            "Open": c * (1 + rng.normal(0, 0.001, n)),
            "High": c * (1 + np.abs(rng.normal(0, 0.004, n))),
            "Low": c * (1 - np.abs(rng.normal(0, 0.004, n))),
            "Close": c,
            "Volume": rng.uniform(100, 1000, n),
        },
        index=idx,
    )


class TestNoLookaheadAudit(unittest.TestCase):
    def test_future_mutation_does_not_change_t_features_or_signal(self):
        original = bars()
        mutated = original.copy()
        t_pos = len(original) - 6
        t = original.index[t_pos]
        mutated.iloc[-5:, mutated.columns.get_indexer(["Open", "High", "Low", "Close"])] = 1e9
        mutated.iloc[-5:, mutated.columns.get_loc("Volume")] = 1e12
        fe1 = CryptoFeatureEngineer()
        a = fe1.compute_features(original, "1h")
        fe2 = CryptoFeatureEngineer()
        b = fe2.compute_features(mutated, "1h")
        pd.testing.assert_series_equal(
            a.loc[t, fe1.feature_columns],
            b.loc[t, fe2.feature_columns],
            check_names=False,
            check_exact=True,
        )
        signal_a = int(a.loc[t, "rsi_14_norm"] > 0)
        signal_b = int(b.loc[t, "rsi_14_norm"] > 0)
        self.assertEqual(signal_a, signal_b)

    def test_taiwan_split_has_five_trading_day_embargo(self):
        universe = {"2330.TW": bars(180, "1D", 1), "2454.TW": bars(180, "1D", 2)}
        ranker = TaiwanStockMLRanker()
        clean, _ = ranker.prepare_dataset(universe)
        train, test = ranker.split_dataset_by_date(clean, embargo_days=5)
        self.assertLess(train.index.max(), test.index.min())
        self.assertEqual(ranker.last_split_info["embargo_trading_days"], 5)
        all_dates = pd.DatetimeIndex(clean.index).normalize().unique().sort_values()
        between = all_dates[(all_dates > train.index.max().normalize()) & (all_dates < test.index.min().normalize())]
        self.assertGreaterEqual(len(between), 5)

    def test_annualization_factors(self):
        self.assertEqual(CryptoFeatureEngineer.get_annualization_factor("15s"), 2102400.0)
        self.assertEqual(CryptoFeatureEngineer.get_annualization_factor("1d"), 365.0)


if __name__ == "__main__":
    unittest.main()
