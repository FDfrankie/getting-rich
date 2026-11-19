"""Unit tests for multi-coin bot execution, per-symbol InstrumentSpec, SQLite disk I/O retry, and Alpha template diversity."""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from decimal import Decimal
from unittest.mock import patch

import numpy as np
import pandas as pd

from crypto_trading.instruments import get_instrument_spec
from crypto_trading.multi_symbol_runner import MultiSymbolBotManager, parse_configured_symbols
from crypto_trading.reconciliation_service import ReconciliationService
from crypto_trading.strategies.momentum_breakout import BaseStrategy
from worldquant_brain.alpha_miner import ContinuousAlphaMiner, extract_expression_skeleton
from worldquant_brain.alpha_template_expander import (
    generate_diverse_domain_candidate,
    list_template_family_count,
)
from worldquant_brain.sqlite_resilience import execute_with_sqlite_retry


class _AlwaysBuyStrategy(BaseStrategy):
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["ATR"] = out["Close"] * 0.01
        out["signal"] = 1.0
        return out


class TestMultiCoinAndAlphaResilience(unittest.TestCase):
    def test_instrument_spec_resolution(self) -> None:
        btc = get_instrument_spec("BTC/USDT")
        eth = get_instrument_spec("ETH/USDT")
        sol = get_instrument_spec("SOLUSDT")
        self.assertEqual(btc.quantity_step, Decimal("0.001"))
        self.assertEqual(eth.price_tick, Decimal("0.01"))
        self.assertEqual(sol.display_symbol, "SOL/USDT")
        self.assertEqual(sol.quantity_step, Decimal("0.01"))
        self.assertEqual(sol.normalize_quantity(1.239), Decimal("1.23"))

    def test_parse_configured_symbols(self) -> None:
        syms = parse_configured_symbols("BTC/USDT, ETHUSDT, sol/usdt, BTC/USDT")
        self.assertEqual(syms, ["BTC/USDT", "ETH/USDT", "SOL/USDT"])

    def test_multicoin_bot_manager_executes_all_coins(self) -> None:
        recon = ReconciliationService()
        recon.reconcile(
            local_positions={},
            remote_positions=[],
            local_orders=[],
            remote_orders=[],
            local_balance=10000.0,
            remote_balance=10000.0,
        )
        mgr = MultiSymbolBotManager(
            symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
            strategy_factory=lambda _sym: _AlwaysBuyStrategy(),
            exchange_adapter=None,
            reconciliation_service=recon,
        )
        # Ensure each bot has reconciliation_service and deterministic synthetic data
        with patch("crypto_trading.regime_filter.HierarchicalRegimeFilter.filter_micro_signal", return_value=1):
            res = mgr.step_all()
        self.assertEqual(set(res.keys()), {"BTC/USDT", "ETH/USDT", "SOL/USDT"})
        for sym, info in res.items():
            self.assertTrue(info["ok"], f"{sym} failed: {info}")
            bot = mgr.get_bot(sym)
            self.assertIsNotNone(bot)
            self.assertGreater(bot.position, 0.0, f"{sym} did not open a position")
            self.assertEqual(bot.position_side, "LONG")

    def test_sqlite_retry_recovers_from_transient_disk_io_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test_retry.db")
            attempts = {"n": 0}

            def _flaky_op(conn: sqlite3.Connection) -> int:
                attempts["n"] += 1
                if attempts["n"] < 3:
                    raise sqlite3.OperationalError("disk I/O error")
                conn.execute("CREATE TABLE IF NOT EXISTS t (x INT)")
                conn.execute("INSERT INTO t VALUES (42)")
                return 42

            val = execute_with_sqlite_retry(db_path, _flaky_op, max_retries=5, base_delay=0.005)
            self.assertEqual(val, 42)
            self.assertEqual(attempts["n"], 3)

    def test_alpha_template_expander_diversity(self) -> None:
        self.assertGreaterEqual(list_template_family_count(), 25)
        skeletons = set()
        for _ in range(120):
            cand = generate_diverse_domain_candidate()
            expr = cand["expression"]
            self.assertNotIn("ts_decay_linear(ts_decay_linear(", expr)
            skeletons.add(extract_expression_skeleton(expr))
        self.assertGreaterEqual(len(skeletons), 18)


if __name__ == "__main__":
    unittest.main()
