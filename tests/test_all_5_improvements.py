"""Offline verification tests for all 5 master improvements."""
from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from crypto_trading.data_loader import CryptoDataLoader, DataSource
from crypto_trading.multi_symbol_runner import MultiSymbolBotManager
from crypto_trading.reconciliation_service import ReconciliationService
from crypto_trading.strategies.momentum_breakout import BaseStrategy
from data_persistence.state_paths import backup_sqlite_to_drive, resolve_sqlite_path
from portfolio.risk_allocator import PortfolioRiskAllocator
from worldquant_brain.brain_auto_submit import (
    REQUIRED_SUBMIT_ACK,
    SUBMIT_ACK_ENV,
    BrainAutoSubmitGate,
)
from worldquant_brain.brain_job_queue import BrainJobQueue, BrainJobStatus, BrainSimulationJob, compute_simulation_key
from worldquant_brain.evidence_repository import (
    EvidenceRepository,
    ExperimentContext,
    OfficialMetrics,
    canonical_json,
)


class _DummyOfflineClient:
    offline_mode = True


class _AlwaysBuyStrategy(BaseStrategy):
    def generate_signals(self, df):
        out = df.copy()
        out["ATR"] = out["Close"] * 0.01
        out["signal"] = 1.0
        return out


class TestAll5Improvements(unittest.TestCase):
    def test_1_brain_auto_submit_and_evidence_repo(self) -> None:
        gate = BrainAutoSubmitGate()
        with patch.dict(os.environ, {SUBMIT_ACK_ENV: REQUIRED_SUBMIT_ACK}):
            ok, reason = gate.can_submit(
                "k1",
                {"sharpe": 1.6, "fitness": 1.2, "turnover": 0.2, "margin": 0.002},
                correlation=0.3,
                alpha_record={
                    "brain_alpha_id": "A1",
                    "metrics_source": "LOCAL_PROXY",
                    "sharpe": 1.6,
                    "fitness": 1.2,
                    "turnover": 0.2,
                },
            )
            self.assertFalse(ok)
            self.assertIn("PRE_SUBMISSION_GATE", reason)

        with tempfile.TemporaryDirectory() as td:
            repo = EvidenceRepository(Path(td) / "ev.db")
            repo.migrate()
            expr = "rank(ebitda / (cap + debt + 0.001))"
            eh = hashlib.sha256(expr.encode("utf-8")).hexdigest()
            settings = {"region": "USA", "universe": "TOP3000", "delay": 1}
            sh = hashlib.sha256(canonical_json(settings).encode("utf-8")).hexdigest()
            repo.create_experiment(
                ExperimentContext(
                    experiment_id="exp_1",
                    candidate_id="cand_1",
                    blueprint_id="bp_1",
                    blueprint_hash=hashlib.sha256(b"bp_1").hexdigest(),
                    hypothesis_family="FUNDAMENTAL_CF_EV",
                    data_domain="FUNDAMENTAL_CF_EV",
                    expression=expr,
                    expression_hash=eh,
                    normalized_expression_hash=eh,
                    structural_expression_hash=eh,
                    settings=settings,
                    settings_hash=sh,
                    region="USA",
                    universe="TOP3000",
                    delay=1,
                    experiment_type="AUTO_MINING",
                    alpha_id="A1",
                )
            )
            repo.record_metrics(
                "exp_1",
                OfficialMetrics(
                    metrics_source="WORLDQUANT_BRAIN",
                    official_simulation_id="sim_1",
                    sharpe=1.55,
                    fitness=1.18,
                    returns=0.12,
                    turnover=0.18,
                    margin=0.0015,
                    submit_ready=True,
                ),
            )
            m = repo.get_metrics("exp_1")
            self.assertEqual(m.official_simulation_id, "sim_1")
            self.assertAlmostEqual(m.sharpe, 1.55)

    def test_2_portfolio_risk_allocator_in_multicoin_manager(self) -> None:
        recon = ReconciliationService()
        recon.reconcile(
            local_positions={},
            remote_positions=[],
            local_orders=[],
            remote_orders=[],
            local_balance=10000.0,
            remote_balance=10000.0,
        )
        alloc = PortfolioRiskAllocator(
            total_equity_usdt=10000.0,
            max_open_positions=2,
            max_total_exposure_pct=1.5,
            max_single_symbol_pct=0.6,
        )
        mgr = MultiSymbolBotManager(
            symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
            strategy_factory=lambda _s: _AlwaysBuyStrategy(),
            exchange_adapter=None,
            reconciliation_service=recon,
            portfolio_allocator=alloc,
        )
        with patch("crypto_trading.regime_filter.HierarchicalRegimeFilter.filter_micro_signal", return_value=1):
            res = mgr.step_all()
        self.assertTrue(res["BTC/USDT"]["ok"])
        self.assertTrue(res["ETH/USDT"]["ok"])
        self.assertTrue(res["SOL/USDT"].get("skipped_by_portfolio_gate"))
        self.assertIn("MAX_OPEN_POSITIONS_REACHED", res["SOL/USDT"].get("reason", ""))

    def test_3_data_loader_local_csv_cache_fallback(self) -> None:
        loader = CryptoDataLoader(symbol="ETH/USDT", timeframe="1h")
        with patch("crypto_trading.data_loader.ccxt", None):
            res = loader.fetch_ohlcv_result(limit=50, allow_synthetic=False, validate=False)
        self.assertEqual(res.source, DataSource.CACHE)
        self.assertFalse(res.is_synthetic)
        self.assertEqual(len(res.df), 50)

    def test_4_sqlite_local_path_and_backup_to_drive(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            local_dir = Path(td) / "local_state"
            drive_backup = Path(td) / "drive_backup"
            resolved = resolve_sqlite_path(
                "UNITTEST_DB_PATH_NONE",
                "test_state.db",
                Path(td) / "G_Drive" / "test_state.db",
                local_state_dir=local_dir,
            )
            self.assertTrue(resolved.startswith(str(local_dir)))
            conn = sqlite3.connect(resolved)
            conn.execute("CREATE TABLE IF NOT EXISTS k (v INT)")
            conn.execute("INSERT INTO k VALUES (99)")
            conn.commit()
            conn.close()

            snap = backup_sqlite_to_drive(resolved, drive_backup)
            self.assertIsNotNone(snap)
            self.assertTrue(Path(snap).exists())
            c2 = sqlite3.connect(snap)
            self.assertEqual(c2.execute("SELECT v FROM k").fetchone()[0], 99)
            c2.close()

    def test_5_brain_job_queue_429_backoff_and_simulation_cache(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "queue.db")
            q = BrainJobQueue(_DummyOfflineClient(), autostart=False, db_path=db_path, num_workers=3)

            self.assertEqual(q.register_rate_limit_429(), 60.0)
            self.assertEqual(q._effective_workers, 1)
            self.assertEqual(q.register_rate_limit_429(), 120.0)
            self.assertEqual(q.register_rate_limit_429(), 240.0)
            self.assertEqual(q.register_rate_limit_429(), 240.0)
            q.clear_rate_limit_429()
            self.assertEqual(q._effective_workers, 3)

            expr = "rank(operating_cashflow / (cap + debt + 0.001))"
            settings = {"region": "USA", "universe": "TOP3000", "delay": 1}
            job = BrainSimulationJob(
                job_id="job_seed_1",
                simulation_key=compute_simulation_key(expr, settings),
                expression=expr,
                settings=settings,
            )
            job.status = BrainJobStatus.COMPLETED
            job.alpha_id = "ALPHA_CACHED_99"
            job.metrics = {"sharpe": 1.88, "fitness": 1.42}
            q._store_simulation_cache(job)

            q2 = BrainJobQueue(_DummyOfflineClient(), autostart=False, db_path=db_path, num_workers=2)
            cached_job, created = q2.create_job(expr, settings=settings)
            self.assertFalse(created)
            self.assertEqual(cached_job.status, BrainJobStatus.COMPLETED)
            self.assertEqual(cached_job.alpha_id, "ALPHA_CACHED_99")


if __name__ == "__main__":
    unittest.main()
