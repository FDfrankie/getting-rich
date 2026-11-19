from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from worldquant_brain.alpha_miner import ContinuousAlphaMiner
from worldquant_brain.brain_job_queue import compute_simulation_key
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


if __name__ == "__main__":
    unittest.main()
