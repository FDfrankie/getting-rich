"""
Mock Tests for P10 Taiwan Stock Research and Governance.
4 tests required by docs/ROUND4_BRAIN_TAIWAN_GOVERNANCE_COMPACT.md:
1. test_taiwan_cross_sectional_ic.py
2. test_scheduler_timezone.py
3. test_report_target_mode.py
4. test_report_escaping.py
"""

from __future__ import annotations

import unittest
from datetime import datetime

from taiwan_stock.taiwan_research import (
    TAIPEI_TZ,
    TaiwanDualSessionScheduler,
    compute_cross_sectional_rank_ic,
    filter_inverse_etfs,
    format_taiwan_report,
)


class TestTaiwanCrossSectionalIC(unittest.TestCase):
    def test_rank_ic_with_adequate_universe(self):
        # 12 stocks
        preds = {f"233{i}.TW": float(i) for i in range(12)}
        # Perfectly correlated returns
        returns = {f"233{i}.TW": float(i) * 0.01 for i in range(12)}

        res = compute_cross_sectional_rank_ic(preds, returns, benchmark_return=0.005)
        self.assertTrue(res["is_official_ic"])
        self.assertEqual(res["target_mode"], "EXCESS_RETURN")
        self.assertAlmostEqual(res["rank_ic"], 1.0, places=2)

    def test_universe_under_10_stocks(self):
        # Only 8 stocks
        preds = {f"233{i}.TW": float(i) for i in range(8)}
        returns = {f"233{i}.TW": float(i) * 0.01 for i in range(8)}

        res = compute_cross_sectional_rank_ic(preds, returns)
        self.assertFalse(res["is_official_ic"])
        self.assertIsNone(res["rank_ic"])
        self.assertEqual(res["universe_size"], 8)
        self.assertIn("Watchlist Ranking, Universe Size=8", res["status_message"])

    def test_inverse_etf_filtering(self):
        symbols = ["2330.TW", "2317.TW", "00632R.TW", "00671R.TW", "2454.TW"]
        filtered = filter_inverse_etfs(symbols)
        self.assertEqual(filtered, ["2330.TW", "2317.TW", "2454.TW"])
        self.assertNotIn("00632R.TW", filtered)


class TestSchedulerTimezone(unittest.TestCase):
    def test_taipei_dual_session(self):
        scheduler = TaiwanDualSessionScheduler()
        self.assertEqual(str(scheduler.tz), "Asia/Taipei")

        # 07:30 pre-market
        dt_0730 = datetime(2026, 11, 16, 7, 30, tzinfo=TAIPEI_TZ)
        s1 = scheduler.determine_session(dt_0730)
        self.assertEqual(s1["session_type"], "PRE_MARKET_WAIT")
        self.assertEqual(s1["data_date_rule"], "PREVIOUS_TRADING_DAY")

        # 09:15 morning session
        dt_0915 = datetime(2026, 11, 16, 9, 15, tzinfo=TAIPEI_TZ)
        s2 = scheduler.determine_session(dt_0915)
        self.assertEqual(s2["session_type"], "MORNING_SESSION")
        self.assertEqual(s2["data_date_rule"], "PREVIOUS_TRADING_DAY")

        # 17:00 afternoon session
        dt_1700 = datetime(2026, 11, 16, 17, 0, tzinfo=TAIPEI_TZ)
        s3 = scheduler.determine_session(dt_1700)
        self.assertEqual(s3["session_type"], "AFTERNOON_SESSION")
        self.assertEqual(s3["data_date_rule"], "CURRENT_TRADING_DAY_COMPLETED")


class TestReportTargetMode(unittest.TestCase):
    def test_target_mode_raw_vs_excess(self):
        preds = {f"stock_{i}": float(i) for i in range(10)}
        rets = {f"stock_{i}": float(i) * 0.02 for i in range(10)}

        # Without benchmark -> RAW_RETURN
        r_raw = compute_cross_sectional_rank_ic(preds, rets, benchmark_return=None)
        self.assertEqual(r_raw["target_mode"], "RAW_RETURN")

        # With benchmark -> EXCESS_RETURN
        r_exc = compute_cross_sectional_rank_ic(preds, rets, benchmark_return=0.01)
        self.assertEqual(r_exc["target_mode"], "EXCESS_RETURN")


class TestReportEscaping(unittest.TestCase):
    def test_escaping_and_currency(self):
        raw_text = "<script>alert('hack');</script> 台股觀察"
        res = format_taiwan_report(raw_text, amount_twd=1500000.0, discipline_score=None)

        # Check escaping
        self.assertNotIn("<script>", res)
        self.assertIn("&lt;script&gt;", res)

        # Check currency format
        self.assertIn("NT$ 1,500,000.00", res)

        # Missing discipline score is N/A, never 100%
        self.assertIn("N/A (無紀錄)", res)
        self.assertNotIn("100%", res)


if __name__ == "__main__":
    unittest.main()
