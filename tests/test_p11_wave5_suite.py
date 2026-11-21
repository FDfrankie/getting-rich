"""Unit tests for P11 Wave 5 & P7/P8 Catch-Up Bundle (Batch 19-22)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_trading.strategies.cross_sectional_crypto_alpha import CrossSectionalCryptoAlpha
from taiwan_stock.fundamental_and_margin import (
    parse_monthly_revenue_rows,
    parse_twse_margin_payload,
    roc_ym_to_pit_effective_date,
)
from us_stock.ml_ranker import USStockMLRanker
from us_stock.sec_edgar_client import parse_company_facts_pit
from worldquant_brain.alpha_template_expander import _TEMPLATE_FAMILIES, _wrap_hygiene
from worldquant_brain.alpha_yield_booster import maybe_invert_negative_sharpe_candidate
from worldquant_brain.two_stage_funnel import STAGE1_PROBES


class TestP11Wave5Suite(unittest.TestCase):
    def test_batch19_orthogonal_probes_templates_and_negative_sharpe_flip(self) -> None:
        probe_names = {name for name, _ in STAGE1_PROBES}
        self.assertIn("reg_neut_cap", probe_names)
        self.assertIn("trade_when_vol", probe_names)

        self.assertIn("OPTIONS_ORTHOGONAL", _TEMPLATE_FAMILIES)
        self.assertIn("FUND_ANALYST_ORTHO", _TEMPLATE_FAMILIES)
        w1 = _wrap_hygiene("close - open", grp="industry", transform="REGRESSION_NEUT_CAP")
        w2 = _wrap_hygiene("close - open", grp="industry", transform="TRADE_WHEN_VOL_EVENT")
        self.assertIn("regression_neut(", w1)
        self.assertIn("trade_when(", w2)

        flipped = maybe_invert_negative_sharpe_candidate("rank(close)", {"sharpe": -1.5, "turnover": 0.25})
        self.assertEqual(flipped, "-1 * (rank(close))")
        self.assertIsNone(maybe_invert_negative_sharpe_candidate("rank(close)", {"sharpe": -0.8, "turnover": 0.25}))

    def test_batch20_cross_sectional_crypto_alpha_dollar_neutral_and_capped(self) -> None:
        dates = pd.date_range("2025-01-01", periods=220, freq="h")
        rng = np.random.default_rng(11)
        btc_ret = rng.normal(0.0002, 0.01, size=len(dates))
        ohlcv_map = {}
        for sym, alpha_drift, fr_base in (
            ("BTC/USDT", 0.0, 0.0001),
            ("ETH/USDT", 0.0010, -0.0002),
            ("SOL/USDT", -0.0010, 0.0005),
            ("BNB/USDT", 0.0004, 0.0000),
            ("AVAX/USDT", -0.0006, 0.0004),
        ):
            r = btc_ret + alpha_drift + rng.normal(0, 0.005, size=len(dates))
            close = 100.0 * np.cumprod(1.0 + r)
            ohlcv_map[sym] = pd.DataFrame(
                {
                    "Close": close,
                    "funding_rate": fr_base + rng.normal(0, 0.00005, size=len(dates)),
                    "open_interest": 1e6 * np.cumprod(1.0 + rng.normal(0.001, 0.01, size=len(dates))),
                },
                index=dates,
            )
        engine = CrossSectionalCryptoAlpha(max_weight_per_asset=0.25)
        weights = engine.compute_target_weights(ohlcv_map)
        self.assertEqual(set(weights.keys()), set(ohlcv_map.keys()))
        for w in weights.values():
            self.assertLessEqual(abs(w), 0.25)
        self.assertTrue(any(w > 0 for w in weights.values()))
        self.assertTrue(any(w < 0 for w in weights.values()))

    def test_batch21_taiwan_monthly_revenue_pit_and_margin_parser(self) -> None:
        self.assertEqual(roc_ym_to_pit_effective_date("11510"), "2026-11-11")
        self.assertEqual(roc_ym_to_pit_effective_date("11512"), "2027-01-11")
        rev = parse_monthly_revenue_rows(
            [
                {
                    "公司代號": "2330",
                    "資料年月": "11510",
                    "營業收入-當月營收": "300,000,000",
                    "營業收入-上月比較增減(%)": "5.2",
                    "營業收入-去年同月增減(%)": "28.4",
                    "累計營業收入-前期比較增減(%)": "31.0",
                }
            ]
        )
        self.assertEqual(rev["2330"]["pit_effective_date"], "2026-11-11")
        self.assertAlmostEqual(rev["2330"]["rev_yoy_pct"], 28.4)

        margin = parse_twse_margin_payload(
            {
                "stat": "OK",
                "tables": [
                    {
                        "fields": ["代號", "名稱", "a", "b", "c", "前日餘額", "今日餘額", "d", "e", "f", "g", "前日券", "今日券"],
                        "data": [["2330", "台積電", "0", "0", "0", "10,000", "9,500", "0", "0", "0", "0", "500", "600"]],
                    }
                ],
            }
        )
        self.assertEqual(margin["2330"]["margin_change_lots"], -500.0)
        self.assertEqual(margin["2330"]["short_change_lots"], 100.0)

    def test_batch22_us_stock_sec_pit_and_brain_screener_wide_pkl_bridge(self) -> None:
        sec_payload = {
            "facts": {
                "us-gaap": {
                    "Assets": {"units": {"USD": [{"form": "10-Q", "filed": "2025-05-01", "val": 1000.0}]}},
                    "GrossProfit": {"units": {"USD": [{"form": "10-Q", "filed": "2025-05-01", "val": 400.0}]}},
                    "OperatingIncomeLoss": {"units": {"USD": [{"form": "10-Q", "filed": "2025-05-01", "val": 200.0}]}},
                    "Revenues": {"units": {"USD": [{"form": "10-Q", "filed": "2025-05-01", "val": 800.0}]}},
                    "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [{"form": "10-Q", "filed": "2025-05-01", "val": 250.0}]}},
                    "NetIncomeLoss": {"units": {"USD": [{"form": "10-Q", "filed": "2025-05-01", "val": 150.0}]}},
                }
            }
        }
        pit_df = parse_company_facts_pit(sec_payload)
        self.assertAlmostEqual(float(pit_df["gross_profitability"].iloc[0]), 0.4)
        self.assertAlmostEqual(float(pit_df["cash_accrual_quality"].iloc[0]), 0.1)

        dates = pd.date_range("2024-01-01", periods=300, freq="B")
        rng = np.random.default_rng(9)
        price_map = {}
        for sym in ("AAPL", "MSFT", "NVDA"):
            close = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.015, size=len(dates)))
            price_map[sym] = pd.DataFrame(
                {
                    "Open": close * 0.995,
                    "High": close * 1.01,
                    "Low": close * 0.99,
                    "Close": close,
                    "Volume": rng.uniform(1e6, 5e6, size=len(dates)),
                },
                index=dates,
            )

        ranker = USStockMLRanker()
        panel = ranker.prepare_panel(price_map, fundamentals_map={"AAPL": pit_df}, sector_map={"AAPL": "TECH", "MSFT": "TECH", "NVDA": "TECH"})
        self.assertIn("sec_rank_mom_12_1", panel.columns)
        self.assertIn("gross_profitability", panel.columns)

        # Verify wide .pkl bridge matches LocalCrossSectionalScreener expected dict keys
        with tempfile.TemporaryDirectory() as td:
            pkl_path = USStockMLRanker.export_panel_for_brain_screener(price_map, Path(td) / "local_us_panel_cache.pkl")
            loaded = pd.read_pickle(pkl_path)
            self.assertIsInstance(loaded, dict)
            for k in ("close", "open", "high", "low", "volume", "returns", "vwap"):
                self.assertIn(k, loaded)
                self.assertEqual(list(loaded[k].columns), ["AAPL", "MSFT", "NVDA"])


if __name__ == "__main__":
    unittest.main()
