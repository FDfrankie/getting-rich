"""Contract & PIT tests for P17 (Wave 10 Institutional Quant Upgrade)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


class TestP17Wave10InstitutionalUpgrade(unittest.TestCase):
    def test_active_pnl_cache_detects_high_local_self_correlation(self) -> None:
        from worldquant_brain.active_pnl_cache import ActivePnlCache

        rng = np.random.default_rng(42)
        base_diff = rng.normal(0.01, 1.0, 60)
        base_cum = np.cumsum(base_diff).tolist()
        clone_cum = np.cumsum(base_diff * 0.99 + rng.normal(0.0, 0.05, 60)).tolist()
        orth_cum = np.cumsum(rng.normal(0.01, 1.0, 60)).tolist()

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            cache = ActivePnlCache(Path(td) / "active_pnl.db")
            cache.upsert_cumulative_pnl("A_LIVE_1", "USA", base_cum)
            corr_high, worst = cache.max_local_self_correlation(clone_cum, region="USA")
            self.assertGreaterEqual(corr_high, 0.90)
            self.assertEqual(worst, "A_LIVE_1")

            corr_low, _ = cache.max_local_self_correlation(orth_cum, region="USA")
            self.assertLess(corr_low, 0.50)

    def test_pick_low_crowding_region_settings(self) -> None:
        from worldquant_brain.alpha_yield_booster import ADVANCED_NEUTRALIZATIONS, pick_low_crowding_region_settings

        s_usa = pick_low_crowding_region_settings(enable_multi_region=False)
        self.assertEqual(s_usa["region"], "USA")
        self.assertIn(s_usa["neutralization"], ADVANCED_NEUTRALIZATIONS)

        s_multi = pick_low_crowding_region_settings(enable_multi_region=True)
        self.assertIn(s_multi["region"], {"USA", "EUR", "ASI", "GLB"})

    def test_binance_algo_order_hedge_mode_gtx_and_risk_event_detection(self) -> None:
        from crypto_trading.order_execution_service import (
            build_binance_futures_order_params,
            is_exchange_risk_event_order,
        )

        ep_stop, p_stop = build_binance_futures_order_params(
            symbol="BTC/USDT", side="SELL", order_type="STOP_MARKET", quantity=0.05, reduce_only=True, hedge_mode=False
        )
        self.assertEqual(ep_stop, "/fapi/v1/algoOrder")
        self.assertEqual(p_stop["reduceOnly"], "true")

        ep_hedge, p_hedge = build_binance_futures_order_params(
            symbol="ETH/USDT", side="SELL", order_type="LIMIT", quantity=1.0, price=3500.0, reduce_only=True, hedge_mode=True, post_only=True
        )
        self.assertEqual(ep_hedge, "/fapi/v1/order")
        self.assertNotIn("reduceOnly", p_hedge)
        self.assertEqual(p_hedge["positionSide"], "LONG")
        self.assertEqual(p_hedge["timeInForce"], "GTX")

        self.assertTrue(is_exchange_risk_event_order("autoclose-123456"))
        self.assertTrue(is_exchange_risk_event_order("adl_autoclose_99"))
        self.assertFalse(is_exchange_risk_event_order("flatten_BTC_123"))

    def test_twse_t86_row_parser_excludes_dealer_hedge_and_converts_to_lots(self) -> None:
        from taiwan_stock.twse_t86 import parse_twse_t86_row

        fields = [
            "證券代號",
            "外陸資買賣超股數(不含外資自營商)",
            "外資自營商買賣超股數",
            "投信買賣超股數",
            "自營商買賣超股數(自行買賣)",
            "自營商買賣超股數(避險)",
            "三大法人買賣超股數",
        ]
        row = ["2330", "1,500,000", "100,000", "400,000", "200,000", "-1,200,000", "1,000,000"]
        parsed = parse_twse_t86_row(fields, row)
        self.assertIsNotNone(parsed)
        self.assertAlmostEqual(parsed["foreign_net_lots"], 1600.0)
        self.assertAlmostEqual(parsed["dealer_prop_lots"], 200.0)
        self.assertAlmostEqual(parsed["dealer_hedge_lots"], -1200.0)
        # smart_institutional_lots = 1600 + 400 + 200 = 2200 (excludes -1200 mechanical hedge!)
        self.assertAlmostEqual(parsed["smart_institutional_lots"], 2200.0)

    def test_us_stock_pead_sue_and_form4_cluster_buy_strictly_pit(self) -> None:
        from us_stock.sec_edgar_client import compute_form4_cluster_buy_score, compute_pead_sue_score

        records = [
            {"end": "2024-03-31", "filed": "2024-05-01", "eps": 1.00},
            {"end": "2024-06-30", "filed": "2024-08-01", "eps": 1.05},
            {"end": "2024-09-30", "filed": "2024-11-01", "eps": 1.10},
            {"end": "2024-12-31", "filed": "2025-02-01", "eps": 1.15},
            {"end": "2025-03-31", "filed": "2025-05-01", "eps": 1.20},
            {"end": "2025-06-30", "filed": "2025-08-01", "eps": 1.50},
            # Future filing after as_of_date must be ignored!
            {"end": "2025-09-30", "filed": "2025-11-05", "eps": 9.99},
        ]
        sue = compute_pead_sue_score(records, as_of_date="2025-11-01")
        self.assertIsNotNone(sue)
        self.assertGreater(sue, 0.0)

        filings = [
            {"filed": "2025-10-15", "transactionCode": "P", "insider_name": "CEO", "shares": 10000, "price": 50.0, "is_10b5_1": False},
            {"filed": "2025-10-20", "transactionCode": "P", "insider_name": "CFO", "shares": 5000, "price": 52.0, "is_10b5_1": False},
            {"filed": "2025-10-22", "transactionCode": "M", "insider_name": "VP", "shares": 50000, "price": 10.0, "is_10b5_1": False},  # option exercise excluded
            {"filed": "2025-11-10", "transactionCode": "P", "insider_name": "DIR", "shares": 9000, "price": 55.0, "is_10b5_1": False},  # future excluded
        ]
        f4 = compute_form4_cluster_buy_score(filings, as_of_date="2025-11-01", lookback_days=30)
        self.assertEqual(f4["distinct_insiders"], 2)
        self.assertGreater(f4["cluster_buy_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
