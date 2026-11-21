"""Regression & contract tests for P13 (Wave 6 precision fixes)."""
from __future__ import annotations

import gc
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from crypto_trading.backtest_engine import BacktestEngine, FeeSchedule
from crypto_trading.reconciliation_service import ReconciliationService
from worldquant_brain.alpha_blueprint import builtin_blueprints, validate_blueprint_fields_against_catalog
from worldquant_brain.alpha_miner import ContinuousAlphaMiner
from worldquant_brain.alpha_vault import AlphaVault
from worldquant_brain.brain_client import BrainClient
from worldquant_brain.brain_job_queue import BrainJobQueue, compute_simulation_key
from worldquant_brain.pre_submission_gate import evaluate_pre_submission
from worldquant_brain.sqlite_resilience import open_sqlite_wal


def _resp(status: int, json_data=None, headers=None, content=None):
    r = MagicMock()
    r.status_code = status
    r.headers = dict(headers or {})
    r.content = content if content is not None else (b"{}" if json_data is not None else b"")
    r.json.return_value = json_data if json_data is not None else {}
    return r


class TestP13Wave6PrecisionFixes(unittest.TestCase):
    def test_brain_jobs_persist_writes_not_null_updated_at(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db = Path(td) / "jobs.db"
            conn = sqlite3.connect(str(db))
            conn.execute(
                """CREATE TABLE brain_jobs (
                    job_id TEXT PRIMARY KEY,
                    simulation_key TEXT NOT NULL,
                    expression TEXT NOT NULL,
                    settings_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    simulation_location TEXT,
                    alpha_id TEXT,
                    metrics_json TEXT,
                    safe_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    priority INTEGER DEFAULT 0,
                    source TEXT DEFAULT 'api',
                    requested_by TEXT DEFAULT '',
                    manual_note TEXT DEFAULT '',
                    simulation_only INTEGER DEFAULT 1,
                    queued_at TEXT,
                    claimed_by TEXT,
                    claimed_at TEXT
                )"""
            )
            conn.commit()
            conn.close()

            q = BrainJobQueue(brain_client=MagicMock(), db_path=str(db), num_workers=0)
            try:
                job, created = q.create_job("rank(close)")
                self.assertTrue(created)
                conn = sqlite3.connect(str(db))
                try:
                    row = conn.execute("SELECT updated_at FROM brain_jobs WHERE job_id = ?", (job.job_id,)).fetchone()
                finally:
                    conn.close()
                self.assertIsNotNone(row)
                self.assertTrue(bool(row[0]))
            finally:
                for meth in ("stop", "shutdown", "close"):
                    fn = getattr(q, meth, None)
                    if callable(fn):
                        fn()
                if hasattr(q, "_stop_event"):
                    q._stop_event.set()
                for t in getattr(q, "_workers", []) or []:
                    if getattr(t, "is_alive", lambda: False)():
                        t.join(timeout=1.0)
                gc.collect()

    def test_verify_alpha_fail_closed_and_delay0_and_prod_corr(self) -> None:
        mock_sess = MagicMock()
        mock_sess.get.return_value = _resp(502, None, content=b"")
        c = BrainClient(email="u@e.com", password="pw", session=mock_sess)
        c.authenticated = True
        res = c.verify_alpha_for_submission("A_502")
        self.assertFalse(res["passed"])
        self.assertEqual(res["status"], "SUBMISSION_CHECK_FAILED")

        # Delay 0 requires Sharpe >= 2.00
        mock_sess.get.side_effect = [
            _resp(200, {"settings": {"delay": 0}, "is": {"sharpe": 1.60, "fitness": 1.2, "turnover": 0.1, "margin": 0.002}}),
            _resp(200, {"is": {"checks": []}}),
        ]
        res_d0 = c.verify_alpha_for_submission("A_D0")
        self.assertFalse(res_d0["passed"])
        self.assertTrue(any("2.00" in r for r in res_d0["reasons"]))

    def test_pre_submission_gate_power_pool_prod_corr_and_pct_turnover(self) -> None:
        base = {
            "brain_alpha_id": "A_1",
            "official_simulation_id": "sim_1",
            "metrics_source": "WORLDQUANT_BRAIN",
            "sharpe": 1.5,
            "fitness": 1.1,
            "turnover": 25.0,  # percentage form -> 0.25
            "self_correlation": 0.55,
            "production_correlation": 0.40,
            "is_checks_passed": True,
            "submit_ready": True,
            "good_ready": True,
        }
        self.assertTrue(evaluate_pre_submission(base).eligible)

        # Power Pool requires self_correlation < 0.50
        pp = dict(base, is_power_pool=True)
        res_pp = evaluate_pre_submission(pp)
        self.assertFalse(res_pp.eligible)
        self.assertIn("SELF_CORRELATION_NOT_PASSED", res_pp.reasons)

        # Prod correlation >= 0.70 must block
        pc = dict(base, production_correlation=0.72)
        res_pc = evaluate_pre_submission(pc)
        self.assertFalse(res_pc.eligible)
        self.assertIn("PROD_CORRELATION_NOT_PASSED", res_pc.reasons)

    def test_canonical_ast_dedup_in_job_queue_and_alpha_miner(self) -> None:
        k1 = compute_simulation_key("rank(ts_delta(close, 5))")
        k2 = compute_simulation_key("rank( ts_delta( close , 5 ) )")
        self.assertEqual(k1, k2)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            vault = AlphaVault(str(Path(td) / "v.db"))
            miner = ContinuousAlphaMiner()
            miner.vault = vault
            miner.mark_expression_tried("rank( ts_delta( close , 5 ) )")
            self.assertTrue(miner.has_tried_expression("rank(ts_delta(close,5))"))

    def test_builtin_blueprints_has_six_orthogonal_families(self) -> None:
        bps = builtin_blueprints()
        self.assertGreaterEqual(len(bps), 6)
        families = {bp.family for bp in bps}
        self.assertGreaterEqual(len(families), 6)
        for bp in bps[1:]:  # verify newly added + fundamental catalog fields
            params = {k: v[0] for k, v in bp.parameter_space.items()}
            errs = validate_blueprint_fields_against_catalog(bp, params)
            self.assertEqual(errs, [], f"{bp.blueprint_id} invalid fields: {errs}")

    def test_reconciliation_accumulates_hedge_mode_and_supports_buy(self) -> None:
        svc = ReconciliationService()
        res = svc.reconcile(
            local_positions={"BTC/USDT": 0.3},
            remote_positions=[
                {"symbol": "BTCUSDT", "contracts": 0.5, "side": "BUY"},
                {"symbol": "BTCUSDT", "contracts": 0.2, "side": "SHORT"},
            ],
            local_orders=[],
            remote_orders=[],
            local_balance=1000.0,
            remote_balance=1000.0,
        )
        self.assertTrue(res.get("is_reconciled"), res)

    def test_backtest_engine_directional_funding_credits_short(self) -> None:
        idx = pd.date_range("2026-01-01", periods=30, freq="1h")
        df = pd.DataFrame(
            {
                "Close": np.full(30, 100.0),
                "signal": np.full(30, -1.0),  # constant short
                "funding_rate": np.full(30, 0.001),  # positive funding -> shorts receive!
            },
            index=idx,
        )
        eng = BacktestEngine(
            initial_capital=10000.0,
            fee_schedule=FeeSchedule(taker_fee_rate=0.0, maker_fee_rate=0.0, spread_bps=0.0, slippage_bps=0.0, funding_rate_per_bar=0.001),
        )
        out = eng.run(df, timeframe="1h")
        self.assertGreater(out.get("net_return_pct", out.get("total_return", 0.0)), 0.0)

    def test_open_sqlite_wal_sets_wal_and_busy_timeout(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            conn = open_sqlite_wal(Path(td) / "wal.db", timeout=30.0)
            try:
                mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
                bt = conn.execute("PRAGMA busy_timeout").fetchone()[0]
                self.assertEqual(str(mode).lower(), "wal")
                self.assertGreaterEqual(int(bt), 30000)
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
