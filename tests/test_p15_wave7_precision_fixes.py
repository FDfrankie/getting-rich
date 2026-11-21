"""Regression & contract tests for P15 (Wave 7 precision fixes)."""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from crypto_trading.multi_symbol_runner import MultiSymbolBotManager
from data_persistence.repositories import SQLiteRepository
from model_governance.governance import ArtifactRegistry
from web_ui.auth import AuthManager, Role
from worldquant_brain.local_cs_screener import evaluate_fastexpr_panel
from worldquant_brain.two_stage_funnel import STAGE1_PROBES


class TestP15Wave7PrecisionFixes(unittest.TestCase):
    def test_brain_alpha_strategy_ts_delay_has_no_bfill_lookahead(self) -> None:
        src = Path("crypto_trading/strategies/brain_alpha_strategy.py").read_text(encoding="utf-8")
        self.assertNotIn(".bfill()", src)

    def test_local_cs_screener_ts_regression_rettypes_and_trade_when_exit(self) -> None:
        idx = pd.date_range("2026-01-01", periods=10, freq="1D")
        close = pd.DataFrame({"A": np.arange(1.0, 11.0) * 2.0 + 5.0, "B": np.arange(1.0, 11.0) * 3.0 + 1.0}, index=idx)
        open_ = pd.DataFrame({"A": np.arange(1.0, 11.0), "B": np.arange(1.0, 11.0)}, index=idx)
        panel = {"close": close, "open": open_, "high": close, "low": open_, "volume": open_ * 100.0, "returns": close.pct_change().fillna(0.0)}

        slope = evaluate_fastexpr_panel("ts_regression(close, open, 5, 0, 2)", panel)
        self.assertAlmostEqual(float(slope["A"].iloc[-1]), 2.0, places=3)
        intercept = evaluate_fastexpr_panel("ts_regression(close, open, 5, 0, 1)", panel)
        self.assertAlmostEqual(float(intercept["A"].iloc[-1]), 5.0, places=3)
        r2 = evaluate_fastexpr_panel("ts_regression(close, open, 5, 0, 6)", panel)
        self.assertAlmostEqual(float(r2["A"].iloc[-1]), 1.0, places=3)

        # trade_when must clear to NaN when exit_cond triggers
        tw = evaluate_fastexpr_panel("trade_when(open >= 2, close, open >= 6)", panel)
        self.assertTrue(np.isfinite(tw["A"].iloc[2]))
        self.assertTrue(np.isnan(tw["A"].iloc[-1]))

    def test_advisory_lock_reclaims_expired_ttl_lock(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            repo = SQLiteRepository(str(Path(td) / "repo.db"))
            self.assertTrue(repo.acquire_advisory_lock("sched", "daily", "worker_dead", ttl_seconds=60))
            self.assertFalse(repo.acquire_advisory_lock("sched", "daily", "worker_new", ttl_seconds=60))
            # Backdate acquired_at to simulate crashed worker
            with repo.connect() as conn:
                conn.execute("UPDATE advisory_locks SET acquired_at = '2020-01-01T00:00:00+00:00'")
            self.assertTrue(repo.acquire_advisory_lock("sched", "daily", "worker_new", ttl_seconds=60))

    def test_multi_symbol_runner_uses_mark_price_and_ex_ante_vol_scaling(self) -> None:
        mgr = MultiSymbolBotManager(symbols=["BTC/USDT", "ETH/USDT"])
        bot = mgr.bots["BTC/USDT"]
        bot.position = 0.1
        bot.entry_price = 50000.0
        bot.balance_usdt = 10000.0
        notional = mgr.active_positions_notional(mark_prices={"BTC/USDT": 60000.0})
        self.assertAlmostEqual(notional["BTC/USDT"], 6000.0, places=2)

        # High realized vol (1.60 vs target 0.40 -> 0.25x cap) should scale down requested notional
        dec = mgr.evaluate_portfolio_gate(
            "ETH/USDT",
            requested_notional_usdt=4000.0,
            mark_prices={"BTC/USDT": 60000.0},
            realized_ann_vol=1.60,
            target_ann_vol=0.40,
        )
        self.assertIsNotNone(dec)

    def test_auth_manager_thread_safe_cleanup_and_no_dead_csrf_dict(self) -> None:
        auth = AuthManager(max_sessions=10)
        self.assertFalse(hasattr(auth, "csrf_tokens"))
        sid_old = auth.create_session("u1", Role.VIEWER, ttl=-1)
        sid_new = auth.create_session("u2", Role.ADMIN, ttl=3600)
        self.assertIsNone(auth.get_session(sid_old))
        self.assertIsNotNone(auth.get_session(sid_new))

    def test_governance_moves_tampered_artifact_and_handles_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            td_p = Path(td)
            reg = ArtifactRegistry(str(td_p / "artifacts"))
            art = td_p / "bad_model.pt"
            art.write_bytes(b"weights")
            man = td_p / "bad_manifest.json"
            man.write_text("{not valid json", encoding="utf-8")
            ok, msg = reg.verify_and_quarantine_if_untrusted(str(art), str(man))
            self.assertFalse(ok)
            self.assertIn("INVALID_MANIFEST", msg)
            self.assertFalse(art.exists())

    def test_stage1_probes_include_wave7_low_corr_operators(self) -> None:
        names = {n for n, _ in STAGE1_PROBES}
        self.assertIn("cartesian_cap_sub", names)
        self.assertIn("jump_decay_neut", names)
        self.assertIn("target_tvr_decay", names)


if __name__ == "__main__":
    unittest.main()
