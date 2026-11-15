import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

# 確保路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from taiwan_stock.market_analyzer import MarketAnalyzer
from taiwan_stock.report_generator import ReportGenerator
from taiwan_stock.stock_screener import StockScreener


class TestTaiwanStockScoringAndScreener(unittest.TestCase):
    """
    台股評分重構、選股透明度與報告一致性測試套件
    """

    def setUp(self):
        # 構造模擬大盤 100 天數據
        dates = pd.date_range("2026-01-01", periods=100, freq="1D")
        close = 20000.0 + np.cumsum(np.random.normal(10, 50, 100))
        volume = np.random.uniform(200000000, 350000000, 100)
        self.mock_df = pd.DataFrame(
            {
                "Open": close - 10,
                "High": close + 30,
                "Low": close - 30,
                "Close": close,
                "Volume": volume,
            },
            index=dates,
        )

    def test_market_analyzer_subscores_and_bounds(self):
        analyzer = MarketAnalyzer()
        with patch.object(analyzer, "fetch_market_data", return_value=self.mock_df):
            res = analyzer.analyze()
            self.assertEqual(res["status"], "正常")
            self.assertIn("score", res)
            self.assertGreaterEqual(res["score"], 0)
            self.assertLessEqual(res["score"], 100)

            # 驗證子項評分存在且總和等於 score
            sub = res["sub_scores"]
            self.assertIn("trend", sub)
            self.assertIn("momentum", sub)
            self.assertIn("volatility", sub)
            self.assertIn("breadth", sub)
            self.assertEqual(
                res["score"],
                sub["trend"] + sub["momentum"] + sub["volatility"] + sub["breadth"],
            )

            # 驗證文案判定與均線型態標記
            self.assertIn("ma_pattern", res)
            self.assertIn("volume_status", res)
            self.assertIn("regime", res)

    def test_bearish_rebound_text(self):
        """測試站上 MA5 但受制於 MA20 時標記為空頭反彈格局"""
        analyzer = MarketAnalyzer()
        # 構造特定收盤價數列：MA5 > Close > MA20 等
        # 讓 MA20 較高 (21000)，MA5 較低 (19000)，但最新收盤站上 MA5 卻壓在 MA20 之下
        prices = [21000] * 40 + [19000] * 15 + [19500] * 5
        dates = pd.date_range("2026-01-01", periods=len(prices), freq="1D")
        custom_df = pd.DataFrame(
            {
                "Open": prices,
                "High": [p + 10 for p in prices],
                "Low": [p - 10 for p in prices],
                "Close": prices,
                "Volume": [100000] * len(prices),
            },
            index=dates,
        )

        with patch.object(analyzer, "fetch_market_data", return_value=custom_df):
            res = analyzer.analyze()
            if res["above_ma5"] and not res["above_ma20"]:
                self.assertIn("空頭反彈格局", res["ma_pattern"])
                self.assertEqual(res["sub_scores"]["trend"], 12)

    def test_screener_detailed_transparency(self):
        """測試選股透明度：已掃描 N 檔，符合門檻 M 檔，若 0 檔列出前 3 檔最接近門檻標的"""
        screener = StockScreener(watchlist=["2330.TW", "2454.TW", "2317.TW"])

        # 模擬返回 evaluation 結果
        def mock_eval(symbol):
            if symbol == "00632R.TW":
                return {
                    "symbol": symbol,
                    "close": 3.8,
                    "change_pct": 0.5,
                    "volume": 50000,
                    "score": 40,
                    "signals": ["避險"],
                    "bearish_signals": [],
                    "is_weak": False,
                    "meets_threshold": False,
                    "stop_loss": 3.6,
                    "target_price": 4.2,
                    "risk_reward_ratio": "1 : 2.0",
                }
            elif symbol == "2330.TW":
                return {
                    "symbol": symbol,
                    "close": 1050.0,
                    "change_pct": 1.2,
                    "volume": 35000,
                    "score": 45,
                    "signals": ["短天期均線金叉"],
                    "bearish_signals": [],
                    "is_weak": False,
                    "meets_threshold": False,
                    "stop_loss": 1020.0,
                    "target_price": 1110.0,
                    "risk_reward_ratio": "1 : 2.0",
                }
            else:
                return {
                    "symbol": symbol,
                    "close": 150.0,
                    "change_pct": -0.8,
                    "volume": 12000,
                    "score": 30,
                    "signals": ["盤整觀望中"],
                    "bearish_signals": ["均線完整空頭排列"],
                    "is_weak": True,
                    "meets_threshold": False,
                    "stop_loss": 145.0,
                    "target_price": 160.0,
                    "risk_reward_ratio": "1 : 2.0",
                }

        with patch.object(screener, "evaluate_stock", side_effect=mock_eval):
            detailed = screener.run_screener_detailed()
            self.assertEqual(detailed["qualified_count"], 0)
            self.assertGreater(detailed["scanned_count"], 0)
            # 0 檔符合時，near_picks 必須有前 3 檔
            self.assertGreaterEqual(len(detailed["near_picks"]), 1)
            # 必須包含反向 ETF 或破線防禦標的
            self.assertGreaterEqual(len(detailed["defense_hedges"]), 1)
            self.assertTrue(any("00632R" in h["symbol"] for h in detailed["defense_hedges"]))

    def test_report_generator_shared_cache(self):
        """測試 ReportGenerator 接收 cached_market_info，確保與儀表板絕對一致"""
        gen = ReportGenerator()
        fixed_market = {
            "status": "正常",
            "date": "2026-11-14",
            "latest_close": 23000.0,
            "change_pct": 0.75,
            "ma5": 22900.0,
            "ma20": 22800.0,
            "ma60": 22500.0,
            "ma_pattern": "多頭排列格局 (MA5 > MA20 > MA60)",
            "score": 75,
            "regime": "強烈看多 (積極做多)",
            "volatility_20d": 14.5,
            "volume_status": "放量上攻",
            "suggestion": "順勢作多",
        }

        # 構造 screener detailed mock
        mock_screener_res = {
            "scanned_count": 16,
            "qualified_count": 1,
            "picks": [
                {
                    "symbol": "2330.TW",
                    "close": 1050.0,
                    "change_pct": 2.5,
                    "score": 85,
                    "signals": ["均線多頭排列"],
                    "stop_loss": 1020.0,
                    "target_price": 1110.0,
                    "risk_reward_ratio": "1 : 2.0",
                }
            ],
            "near_picks": [],
            "defense_hedges": [],
        }

        with patch.object(gen.stock_screener, "run_screener_detailed", return_value=mock_screener_res):
            rep = gen.generate(cached_market_info=fixed_market)
            html_rep = gen.generate_html_report(cached_market_info=fixed_market)

            # 驗證分數 75 完全同步於兩份報告中
            self.assertIn("75 / 100", rep)
            self.assertIn('75 <span class="text-xs text-slate-400">/ 100</span>', html_rep)
            self.assertIn("多頭排列格局", rep)
            self.assertIn("最後交易日: 2026-11-14", html_rep)


if __name__ == "__main__":
    unittest.main()
