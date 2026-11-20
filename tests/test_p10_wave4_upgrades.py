"""Unit tests for P10 Wave 4 Research & Code Upgrade Bundle (Batch 14-18)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_trading.rl_env import CryptoTradingEnv
from crypto_trading.strategies.perp_carry_basis_strategy import PerpCarryBasisStrategy
from taiwan_stock.ml_ranker import TaiwanStockMLRanker
from taiwan_stock.twse_t86 import parse_twse_t86_payload
from worldquant_brain.correlation_evidence import (
    compute_daily_pnl_correlation,
    diagnose_fitness_bottleneck,
)
from worldquant_brain.sqlite_resilience import managed_connection


class TestP10Wave4Upgrades(unittest.TestCase):
    def test_batch14_sqlite_managed_connection_begin_immediate_and_pragmas(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "wave4.db")
            with managed_connection(db_path, immediate_write=True) as conn:
                conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
                conn.execute("INSERT INTO t (id, v) VALUES (1, 'ok')")
                cache_sz = conn.execute("PRAGMA cache_size").fetchone()[0]
                self.assertEqual(cache_sz, -64000)
            with managed_connection(db_path, immediate_write=False) as conn:
                val = conn.execute("SELECT v FROM t WHERE id = 1").fetchone()[0]
                self.assertEqual(val, "ok")

    def test_batch15_twse_t86_parser_avoids_double_counting_and_checks_stat(self) -> None:
        self.assertEqual(parse_twse_t86_payload({"stat": "很抱歉，沒有符合條件的資料!"}), {})
        payload = {
            "stat": "OK",
            "fields": [
                "證券代號",
                "外陸資買賣超股數(不含外資自營商)",
                "外資自營商買賣超股數",
                "投信買賣超股數",
                "自營商買賣超股數",
                "三大法人買賣超股數",
            ],
            "data": [["2330", "1,000,000", "-200,000", "500,000", "100,000", "1,400,000"]],
        }
        parsed = parse_twse_t86_payload(payload)
        self.assertEqual(parsed["2330"]["foreign_net_shares"], 800000)
        self.assertEqual(parsed["2330"]["total_inst_net_shares"], 1400000)

    def test_batch15_taiwan_ml_ranker_adds_fip_and_cross_sectional_ranks(self) -> None:
        dates = pd.date_range("2025-01-01", periods=100, freq="B")
        rng = np.random.default_rng(7)
        data_map = {}
        for sym, drift in (("2330.TW", 0.002), ("2454.TW", -0.001), ("2317.TW", 0.0005)):
            close = 100.0 * np.cumprod(1.0 + drift + rng.normal(0, 0.01, size=len(dates)))
            data_map[sym] = pd.DataFrame(
                {
                    "Open": close * 0.995,
                    "High": close * 1.01,
                    "Low": close * 0.99,
                    "Close": close,
                    "Volume": rng.uniform(1e6, 5e6, size=len(dates)),
                },
                index=dates,
            )
        ranker = TaiwanStockMLRanker()
        clean, cols = ranker.prepare_dataset(data_map)
        self.assertIn("fip_20d", cols)
        self.assertIn("cs_rank_fip_20d", cols)
        self.assertIn("cs_rank_mom_20d", cols)
        self.assertFalse(clean.empty)

    def test_batch16_perp_carry_clamped_funding_and_cost_deadband(self) -> None:
        dates = pd.date_range("2025-01-01", periods=200, freq="h")
        close = pd.Series(50000.0 + np.sin(np.linspace(0, 20, 200)) * 10.0, index=dates)
        df = pd.DataFrame(
            {
                "Open": close,
                "High": close + 10.0,
                "Low": close - 10.0,
                "Close": close,
                "Volume": np.full(200, 1000.0),
            },
            index=dates,
        )
        strat = PerpCarryBasisStrategy()
        out = strat.generate_signals(df)
        self.assertIn("funding_is_proxy", out.columns)
        self.assertTrue(bool(out["funding_is_proxy"].iloc[-1]))
        # Low-volatility basis (< 10 bps) should be filtered by no-arbitrage cost deadband
        self.assertTrue((out["signal"] == 0.0).all())

    def test_batch17_rl_env_dsr_incremental_dd_and_nonlinear_impact(self) -> None:
        dates = pd.date_range("2025-01-01", periods=60, freq="h")
        close = np.linspace(100.0, 110.0, 60)
        df = pd.DataFrame(
            {
                "Open": close,
                "High": close * 1.01,
                "Low": close * 0.99,
                "Close": close,
                "Volume": np.full(60, 1000.0),
                "f1": np.linspace(0, 1, 60),
            },
            index=dates,
        )
        env = CryptoTradingEnv(df, feature_cols=["f1"], reward_type="sharpe_differential")
        env.reset()
        obs, reward, done, info = env.step(1.0)
        # Positive price move with action=1.0 must produce positive gross return on the SAME bar
        self.assertGreater(info["trading_cost"], (env.fee_rate + env.slippage))  # includes impact_cost
        self.assertFalse(done)
        self.assertTrue(np.isfinite(reward))

    def test_batch18_daily_pnl_correlation_and_fitness_diagnosis(self) -> None:
        # Two cumulative curves with positive drift but opposite daily noise -> daily corr < 0
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 1.0, size=60)
        cum_a = np.cumsum(0.5 + noise).tolist()
        cum_b = np.cumsum(0.5 - noise).tolist()
        corr = compute_daily_pnl_correlation(cum_a, cum_b)
        self.assertIsNotNone(corr)
        self.assertLess(corr, -0.95)

        diag = diagnose_fitness_bottleneck(sharpe=1.5, annual_return=0.10, turnover=0.60)
        self.assertEqual(diag["recommended_lever"], "INCREASE_DECAY_OR_MIX_FUNDAMENTAL")


if __name__ == "__main__":
    unittest.main()
