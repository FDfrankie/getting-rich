"""
Tests for causal cross-asset alignment: ensuring no backward fill (bfill)
and that future data does not leak into the past.
"""

import numpy as np
import pandas as pd

from crypto_trading.feature_engineer import CryptoFeatureEngineer


def test_no_bfill_lookahead_leak():
    # Construct two asset dataframes where Asset B starts later than Asset A
    dates = pd.date_range("2026-01-01", periods=10, freq="1h")
    df_a = pd.DataFrame(
        {
            "Open": [100.0] * 10,
            "High": [105.0] * 10,
            "Low": [95.0] * 10,
            "Close": [100.0 + i for i in range(10)],
            "Volume": [1000.0] * 10,
        },
        index=dates,
    )

    # Asset B only starts at index 3
    df_b = pd.DataFrame(
        {
            "Open": [50.0] * 7,
            "High": [55.0] * 7,
            "Low": [45.0] * 7,
            "Close": [50.0 + i for i in range(7)],
            "Volume": [500.0] * 7,
        },
        index=dates[3:],
    )

    fe = CryptoFeatureEngineer()
    cross_df = fe.compute_multi_asset_cross_features({"BTC/USDT": df_a, "ETH/USDT": df_b})

    # At indices 0, 1, 2, ETH was not available yet.
    # If bfill had occurred, ETH's index 3 price (50.0) would leak to 0, 1, 2.
    # Without bfill, ETH close is NaN, and correlation / spread shouldn't have lookahead values.
    assert pd.isna(cross_df.iloc[0]["ETH_corr_BTC_24"]) or cross_df.iloc[0]["ETH_corr_BTC_24"] == 0.0


def test_target_direction_tail_is_pd_na():
    dates = pd.date_range("2026-01-01", periods=15, freq="1h")
    df = pd.DataFrame(
        {
            "Open": [100.0] * 15,
            "High": [105.0] * 15,
            "Low": [95.0] * 15,
            "Close": [100.0 + i for i in range(15)],
            "Volume": [1000.0] * 15,
        },
        index=dates,
    )

    fe = CryptoFeatureEngineer(forward_horizons=[1, 3])
    feats = fe.compute_features(df)

    # For horizon=1, the last row's target_fwd_ret_1 is NaN, and target_direction_1 MUST be pd.NA
    assert pd.isna(feats["target_fwd_ret_1"].iloc[-1])
    assert pd.isna(feats["target_direction_1"].iloc[-1])

    # For horizon=3, the last 3 rows MUST be pd.NA
    for idx in (-1, -2, -3):
        assert pd.isna(feats["target_fwd_ret_3"].iloc[idx])
        assert pd.isna(feats["target_direction_3"].iloc[idx])


def test_obv_slope_10_is_rolling_ratio():
    dates = pd.date_range("2026-01-01", periods=20, freq="1h")
    # All increasing prices -> diff > 0 -> direction = +1
    df = pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(20)],
            "High": [105.0 + i for i in range(20)],
            "Low": [95.0 + i for i in range(20)],
            "Close": [100.0 + i for i in range(20)],
            "Volume": [100.0] * 20,
        },
        index=dates,
    )

    fe = CryptoFeatureEngineer()
    feats = fe.compute_features(df)
    # Once rolled 10 periods, net active vol = 10 * 100, tot vol = 10 * 100 -> ratio = +1.0
    val = feats["obv_slope_10"].iloc[-1]
    assert np.isclose(val, 1.0)
