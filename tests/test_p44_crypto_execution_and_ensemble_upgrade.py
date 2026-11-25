"""P44 unit tests: -4003/-2019 execution hardening, NativeCryptoEnsembleStrategy causality & OOS edge,
TaiwanStockMLRanker cs_rank inference, and synthetic DB row hygiene.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from crypto_trading.bot_runner import CryptoTradingBot
from crypto_trading.db_hygiene import purge_synthetic_test_rows
from crypto_trading.order_execution_service import build_binance_futures_order_params
from crypto_trading.order_manager import OrderManager, OrderState
from crypto_trading.risk_manager import DecisionAction, RiskManager
from crypto_trading.strategies.momentum_breakout import BaseStrategy, MomentumBreakoutStrategy
from crypto_trading.strategies.native_ensemble_strategy import NativeCryptoEnsembleStrategy
from data_persistence.gdrive_sync import LocalPersistenceService
from taiwan_stock.ml_ranker import TaiwanStockMLRanker


class _FixedSignalStrategy(BaseStrategy):
    def __init__(self, sig: float = 1.0) -> None:
        self._sig = sig

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["ATR"] = out["Close"] * 0.01
        out["signal"] = self._sig
        return out


class _MockExchangeClient:
    is_connected = True
    mock_mode = False

    def __init__(self, avg_price: float = 61000.0) -> None:
        self.calls: list[dict] = []
        self.avg_price = avg_price

    def create_order(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "success": True,
            "exchange_order_id": "EX_12345",
            "order": {"orderId": "EX_12345", "avgPrice": str(self.avg_price), "status": "closed"},
        }


class TestP44CryptoExecutionAndEnsemble(unittest.TestCase):
    def test_4003_non_bypassable_zero_and_negative_qty_guardrails(self) -> None:
        rm = RiskManager(initial_capital=10000.0)
        ctx = {
            "data_source": "test",
            "data_quality_passed": True,
            "reconciliation_fresh": True,
            "exchange_connected": True,
            "step_size": 0.001,
            "min_quantity": 0.001,
            "tick_size": 0.1,
            "min_notional": 5.0,
        }
        for bad_qty in (0.0, -0.081, 0.0001):
            for reduce_only in (False, True):
                dec = rm.evaluate_order("BTC/USDT", "BUY", "MARKET", bad_qty, 60000.0, is_reduce_only=reduce_only, context=ctx)
                self.assertFalse(dec.allowed, f"RiskManager must reject qty={bad_qty} reduce_only={reduce_only}")
                self.assertEqual(dec.action, DecisionAction.REJECT)

        with tempfile.TemporaryDirectory() as td:
            om = OrderManager(db_path=os.path.join(td, "om.db"))
            client = _MockExchangeClient()
            for bad_qty in (0.0, -0.05):
                res = om.execute_with_idempotency(client, "BTC/USDT", "BUY", "MARKET", bad_qty, 60000.0, reduce_only=True)
                self.assertFalse(res["success"])
                self.assertEqual(res["state"], OrderState.REJECTED.value)
            self.assertEqual(len(client.calls), 0, "Exchange client must never be called for qty <= 0")

        with self.assertRaises(ValueError):
            build_binance_futures_order_params(symbol="BTCUSDT", side="BUY", order_type="MARKET", quantity=0.0)
        with self.assertRaises(ValueError):
            build_binance_futures_order_params(symbol="BTCUSDT", side="BUY", order_type="MARKET", quantity=-0.081)

    def test_bot_auto_heals_zero_position_and_passes_reduce_only_and_avg_price(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp_db = os.path.join(td, "bot.db")
            client = _MockExchangeClient(avg_price=61234.5)
            with (
                patch("crypto_trading.bot_runner.OrderManager", side_effect=lambda *a, **kw: OrderManager(db_path=tmp_db)),
                patch("crypto_trading.bot_runner.LocalPersistenceService", side_effect=lambda *a, **kw: LocalPersistenceService(db_path=tmp_db)),
            ):
                bot = CryptoTradingBot(symbol="BTC/USDT", strategy=_FixedSignalStrategy(0.0), exchange_adapter=client)
                bot.reconciliation_service = MagicMock(is_ready_for_trading=lambda: True)
                # Simulate stale SHORT side with 0.0 position (the exact -4003 trigger)
                bot.position_side = "SHORT"
                bot.position = 0.0
                bot.execute_order("BUY", bot.position, 60000.0, reason="exit", reduce_only=True)
                self.assertEqual(bot.position_side, "FLAT")
                self.assertEqual(len(client.calls), 0)

                # Now simulate real LONG position close -> must pass reduce_only=True and use avgPrice
                bot.position_side = "LONG"
                bot.position = 0.05
                bot.entry_price = 60000.0
                bot.execute_order("SELL", 0.05, 60500.0, reason="exit", reduce_only=True)
                self.assertEqual(len(client.calls), 1)
                self.assertTrue(client.calls[0]["reduce_only"])
                self.assertEqual(bot.position_side, "FLAT")
                logs = bot.sync.get_trade_logs(limit=5)
                self.assertEqual(logs[0]["action"], "CLOSE")
                self.assertAlmostEqual(logs[0]["price"], 61234.5, places=2)
                self.assertGreater(logs[0]["pnl"], 0.0)

    def test_2019_margin_and_notional_cap_in_bot_step(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp_db = os.path.join(td, "cap.db")
            with (
                patch("crypto_trading.bot_runner.OrderManager", side_effect=lambda *a, **kw: OrderManager(db_path=tmp_db)),
                patch("crypto_trading.bot_runner.LocalPersistenceService", side_effect=lambda *a, **kw: LocalPersistenceService(db_path=tmp_db)),
                patch("crypto_trading.regime_filter.HierarchicalRegimeFilter.filter_micro_signal", return_value=1),
            ):
                bot = CryptoTradingBot(symbol="BTC/USDT", strategy=_FixedSignalStrategy(1.0), exchange_adapter=None)
                bot.step(max_notional_usdt=1500.0, free_balance_usdt=1200.0)
                notional = bot.position * bot.entry_price
                self.assertGreater(bot.position, 0.0)
                self.assertLessEqual(notional, 1200.0 * 0.90 + 1e-6)

    def test_native_ensemble_strictly_causal_and_beats_momentum_oos(self) -> None:
        cache_dir = Path(__file__).resolve().parent.parent / "crypto_trading" / "data_cache"
        btc_csv = cache_dir / "BTC_USDT_1h_180d.csv"
        if not btc_csv.exists():
            self.skipTest("BTC_USDT_1h_180d.csv not present")
        df = pd.read_csv(btc_csv, index_col=0, parse_dates=True).sort_index()

        # 1. Strict causality check: perturbing future bars after idx=500 must NOT alter signals[:500]
        sub = df.iloc[:700].copy()
        sub_mod = sub.copy()
        sub_mod.iloc[501:, sub_mod.columns.get_loc("Close")] *= 1.35
        sub_mod.iloc[501:, sub_mod.columns.get_loc("High")] *= 1.40
        sub_mod.iloc[501:, sub_mod.columns.get_loc("Low")] *= 0.70

        s1 = NativeCryptoEnsembleStrategy().generate_signals(sub)["signal"].iloc[:500].to_numpy()
        s2 = NativeCryptoEnsembleStrategy().generate_signals(sub_mod)["signal"].iloc[:500].to_numpy()
        np.testing.assert_array_equal(s1, s2)

        # 2. OOS comparison vs MomentumBreakoutStrategy on last 30% of cached 1h coins
        ens_rets, mom_rets = [], []
        for fname in ("BTC_USDT_1h_180d.csv", "ETH_USDT_1h_180d.csv", "SOL_USDT_1h_180d.csv"):
            p = cache_dir / fname
            if not p.exists():
                continue
            d = pd.read_csv(p, index_col=0, parse_dates=True).sort_index()
            split = int(len(d) * 0.70)
            for strat, bucket in ((NativeCryptoEnsembleStrategy(), ens_rets), (MomentumBreakoutStrategy(), mom_rets)):
                out = strat.generate_signals(d).iloc[split:]
                sig = out["signal"].fillna(0.0).to_numpy(float)
                ret1 = out["Close"].pct_change().shift(-1).fillna(0.0).to_numpy(float)
                turn = np.abs(sig - np.concatenate([[0.0], sig[:-1]]))
                net = sig * ret1 - turn * 0.0007
                eq = float(np.cumprod(1.0 + np.clip(net, -0.95, 5.0))[-1] - 1.0)
                bucket.append(eq)
        self.assertGreater(float(np.mean(ens_rets)), float(np.mean(mom_rets)))

    def test_taiwan_ranker_predict_rankings_computes_cs_rank(self) -> None:
        np.random.seed(7)
        dates = pd.date_range("2025-01-01", periods=260, freq="B")
        uni = {}
        for sym, drift in (("2330.TW", 0.0005), ("2454.TW", -0.0002), ("2317.TW", 0.0001)):
            c = 100.0 * np.exp(np.cumsum(np.random.normal(drift, 0.015, size=len(dates))))
            uni[sym] = pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": np.full(len(dates), 1e5)}, index=dates)
        ranker = TaiwanStockMLRanker(model_type="lightgbm")
        ds, _ = ranker.prepare_dataset(uni)
        ranker.train_model(ds)
        res = ranker.predict_rankings(uni)
        self.assertEqual(len(res), 3)
        self.assertIn("predicted_fwd_5d_ret_pct", res.columns)

    def test_purge_synthetic_test_rows_preserves_real_trades(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "hygiene.db")
            svc = LocalPersistenceService(db_path=db)
            svc.record_trade_log("BTC/USDT", "OPEN", "BUY", 50000.0, 1.0, reason="Open Long")
            svc.record_trade_log("BTC/USDT", "CLOSE", "SELL", 55000.0, 1.0, pnl=5000.0, pnl_pct=10.0, reason="Close Long")
            svc.record_trade_log("BTC/USDT", "CLOSE", "BUY", 58418.0, 0.034, pnl=10.73, pnl_pct=0.54, reason="策略離場信號")
            svc.record_order("BTC/USDT", "BUY", "MARKET", 65000.0, 0.01, "FILLED", exchange_order_id="987654321", reason="unit_test")
            svc.record_order("BTC/USDT", "SELL", "MARKET", 58418.0, 0.034, "FILLED", exchange_order_id="1231708823", reason="策略離場信號")

            res = purge_synthetic_test_rows(db_path=db, dry_run=False, backup_dir=os.path.join(td, "bk"))
            self.assertEqual(res["trade_logs_deleted"], 2)
            self.assertEqual(res["orders_deleted"], 1)
            self.assertEqual(len(svc.get_trade_logs(limit=10)), 1)
            self.assertEqual(len(svc.get_orders(limit=10)), 1)


if __name__ == "__main__":
    unittest.main()
