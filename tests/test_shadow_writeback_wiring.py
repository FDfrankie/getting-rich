from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

from worldquant_brain.alpha_miner import ContinuousAlphaMiner
from worldquant_brain.alpha_repository import AlphaRepository
from worldquant_brain.alpha_vault import AlphaVault
from worldquant_brain.brain_job_queue import BrainJobQueue, BrainJobStatus, BrainSimulationJob, compute_simulation_key
from worldquant_brain.brain_scheduler import BrainDailyScheduler
from worldquant_brain.shadow_writeback import record_simulation_shadow


class TestShadowWritebackWiring(unittest.TestCase):
    def test_record_simulation_shadow_persists_pareto_and_motif(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db_path = Path(td) / "shadow_test.db"
            res = record_simulation_shadow(
                db_path,
                expression="group_neutralize(-1 * ts_std_dev(ts_backfill(rank(scl12_buzz), 5), 7), subindustry)",
                parent_expression="group_neutralize(-1 * ts_std_dev(ts_backfill(scl12_buzz, 5), 7), subindustry)",
                settings={"region": "USA", "delay": 1, "universe": "TOP3000"},
                metrics={"sharpe": 1.42, "fitness": 1.10, "turnover": 0.14, "checks_passed": False, "failed_checks": ["LOW_SUB_UNIVERSE_SHARPE"]},
                alpha_id="alpha_shadow_1",
                dataset_id="sentiment",
            )
            self.assertFalse(res["degraded"])
            self.assertTrue(res["recorded_pareto"])
            self.assertTrue(res["recorded_motif"])

            conn = sqlite3.connect(str(db_path))
            try:
                p_cnt = conn.execute("SELECT COUNT(*) FROM pareto_archive_entries").fetchone()[0]
                m_cnt = conn.execute("SELECT COUNT(*) FROM research_edit_motifs").fetchone()[0]
                c_cnt = conn.execute("SELECT COUNT(*) FROM research_motif_contexts").fetchone()[0]
                self.assertEqual(p_cnt, 1)
                self.assertEqual(m_cnt, 1)
                self.assertEqual(c_cnt, 1)
            finally:
                conn.close()

    def test_record_simulation_shadow_degrades_gracefully_on_bad_path(self) -> None:
        bad_path = Path("Z:/nonexistent_drive_xyz/cannot_write/bad.db")
        res = record_simulation_shadow(
            bad_path,
            expression="rank(close)",
            metrics={"sharpe": 1.1, "fitness": 0.9},
        )
        self.assertTrue(res["degraded"])
        self.assertFalse(res["recorded_pareto"])
        self.assertIn("error", res)

    def test_compute_simulation_key_uses_canonical_ast(self) -> None:
        k1 = compute_simulation_key("rank( ts_delta( close , 5 ) )", {"delay": 1})
        k2 = compute_simulation_key("rank(ts_delta(close,5))", {"delay": 1})
        self.assertEqual(k1, k2)

    def test_miner_tried_dedup_and_shadow_outcome(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db_path = Path(td) / "miner_test.db"
            miner = ContinuousAlphaMiner()
            miner.vault.db_path = str(db_path)
            miner.mark_expression_tried("rank( ts_delta( close , 5 ) )")
            self.assertTrue(miner.has_tried_expression("rank(ts_delta(close,5))"))

            out = miner.record_simulation_outcome(
                expression="rank(ts_delta(close, 5))",
                settings={"region": "USA", "delay": 1, "universe": "TOP3000"},
                metrics={"sharpe": 1.35, "fitness": 1.05, "turnover": 0.18, "checks_passed": True},
                dataset_id="pv1",
            )
            self.assertTrue(out["recorded_pareto"])

    def test_scheduler_finish_records_shadow_safely(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db_path = Path(td) / "sched_test.db"
            sched = BrainDailyScheduler(db_path, flush_hour=0)
            now = datetime(2026, 11, 20, 23, 0, tzinfo=timezone.utc)
            claim = sched.claim_due_run(worker_id="w1", now=now)
            self.assertTrue(claim.acquired)
            sched.finish(
                claim.run_id,
                status="COMPLETED",
                considered=5,
                submitted=1,
                now=now,
                shadow_expression="rank(close)",
                shadow_settings={"region": "USA", "delay": 1, "universe": "TOP3000"},
                shadow_metrics={"sharpe": 1.5, "fitness": 1.2, "turnover": 0.12},
            )
            conn = sqlite3.connect(str(db_path))
            try:
                p_cnt = conn.execute("SELECT COUNT(*) FROM pareto_archive_entries").fetchone()[0]
                self.assertEqual(p_cnt, 1)
            finally:
                conn.close()

    def test_record_simulation_shadow_ignores_nan_metrics(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db_path = Path(td) / "nan_test.db"
            res = record_simulation_shadow(
                db_path,
                expression="rank(close)",
                metrics={"sharpe": float("nan"), "fitness": 1.0, "turnover": float("nan")},
            )
            self.assertFalse(res["degraded"])
            self.assertTrue(res["recorded_pareto"])
            conn = sqlite3.connect(str(db_path))
            try:
                row = conn.execute("SELECT official_quality, cost_efficiency FROM pareto_archive_entries").fetchone()
                self.assertIsNone(row[0])
                self.assertIsNone(row[1])
            finally:
                conn.close()

    def test_alpha_vault_and_repository_use_canonical_expression_hash(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            vault = AlphaVault(str(Path(td) / "vault.db"))
            h1 = vault._canonical_expression_hash("rank( ts_delta( close , 5 ) )")
            h2 = vault._canonical_expression_hash("rank(ts_delta(close,5))")
            self.assertEqual(h1, h2)

            repo = AlphaRepository(Path(td) / "repo.db")
            r1 = repo.save_candidate({"alpha_id": "a1", "expression": "rank( ts_delta( close , 5 ) )"})
            r2 = repo.save_candidate({"alpha_id": "a2", "expression": "rank(ts_delta(close,5))"})
            self.assertEqual(r1["expression_hash"], r2["expression_hash"])

    def test_brain_job_queue_execute_job_records_shadow_and_cache(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db_path = Path(td) / "queue_test.db"
            mock_client = MagicMock()
            mock_client.offline_mode = False
            mock_client.authenticated = True
            mock_client.BASE_URL = "https://api.worldquantbrain.com"
            mock_client.timeout = 5.0
            mock_client._number.side_effect = lambda src, key: src.get(key)

            post_resp = MagicMock()
            post_resp.status_code = 201
            post_resp.headers = {"Location": "https://api.worldquantbrain.com/simulations/sim_123"}

            poll_resp = MagicMock()
            poll_resp.status_code = 200
            poll_resp.headers = {}
            poll_resp.json.return_value = {"status": "COMPLETE", "alpha": "alpha_999"}

            alpha_resp = MagicMock()
            alpha_resp.status_code = 200
            alpha_resp.json.return_value = {
                "id": "alpha_999",
                "is": {
                    "sharpe": 1.55,
                    "fitness": 1.25,
                    "turnover": 0.14,
                    "margin": 0.0018,
                    "checks": [{"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "PASS"}],
                },
            }
            mock_client.session.post.return_value = post_resp
            mock_client.session.get.side_effect = [poll_resp, alpha_resp]

            queue = BrainJobQueue(brain_client=mock_client, db_path=db_path, autostart=False)
            settings = {"region": "USA", "delay": 1, "universe": "TOP3000"}
            expr = "rank(ts_delta(close, 5))"
            job = BrainSimulationJob(
                job_id="job_1",
                simulation_key=compute_simulation_key(expr, settings),
                expression=expr,
                settings=settings,
            )
            queue._persist_job(job)
            queue._execute_job(job)

            self.assertEqual(job.status, BrainJobStatus.COMPLETED)
            self.assertEqual(job.alpha_id, "alpha_999")
            self.assertAlmostEqual(job.metrics["sharpe"], 1.55)

            conn = sqlite3.connect(str(db_path))
            try:
                cache_cnt = conn.execute("SELECT COUNT(*) FROM brain_simulation_cache").fetchone()[0]
                pareto_cnt = conn.execute("SELECT COUNT(*) FROM pareto_archive_entries").fetchone()[0]
                self.assertEqual(cache_cnt, 1)
                self.assertEqual(pareto_cnt, 1)
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
