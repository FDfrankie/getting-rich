"""Unit tests for P18 Funnel False-Failure Recovery & Consecutive-Failure Circuit Breaker."""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from worldquant_brain.two_stage_funnel import TwoStageAlphaFunnel


class TestP18FunnelRecoveryAndCircuitBreaker(unittest.TestCase):
    def test_unsimulated_api_failure_releases_funnel_tried_and_skips_rl_penalty(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db = Path(td) / "vault.db"
            funnel = TwoStageAlphaFunnel(str(db))
            ds_fields = {"fundamental6": ["fnd6_ebitda", "fnd6_sales"]}
            batch = funnel.next_batch(ds_fields, batch_size=2)
            self.assertEqual(len(batch), 2)

            # Simulate transient 401/timeout API failure (success=False, sharpe=None)
            failed_results = [
                {
                    "success": False,
                    "status": "ERROR",
                    "error": "HTTP 401 session expired",
                    "expression": b["expression"],
                    "settings": b["settings"],
                    "stage": b["stage"],
                    "dataset_id": b["dataset_id"],
                    "probe": b.get("probe", ""),
                }
                for b in batch
            ]
            ing = funnel.ingest_results(failed_results)
            self.assertEqual(ing["unsimulated_released"], 2)
            self.assertEqual(ing["rejected"], 0)

            # Those 2 formulas must be released from funnel_tried (count == 0)
            conn = sqlite3.connect(str(db))
            try:
                tried_cnt = conn.execute("SELECT COUNT(*) FROM funnel_tried").fetchone()[0]
                rl_cnt = conn.execute("SELECT COUNT(*) FROM funnel_rl_policy").fetchone()[0]
            finally:
                conn.close()
            self.assertEqual(tried_cnt, 0)
            self.assertEqual(rl_cnt, 0)

    def test_purge_false_failures_cleans_null_sharpe_but_preserves_real_simulations(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db = Path(td) / "vault.db"
            funnel = TwoStageAlphaFunnel(str(db))
            conn = sqlite3.connect(str(db))
            try:
                # 1 real simulated rejection (sharpe = 0.42) + 2 false unsimulated rejections (sharpe = NULL)
                conn.execute("INSERT INTO funnel_tried VALUES ('h_real', 'rank(close)', '{}', 'STAGE1', 'pv1', '2026-11-21T00:00:00Z')")
                conn.execute("INSERT INTO funnel_tried VALUES ('h_fake1', 'rank(open)', '{}', 'STAGE1', 'analyst4', '2026-11-21T00:00:00Z')")
                conn.execute("INSERT INTO funnel_tried VALUES ('h_fake2', 'rank(high)', '{}', 'STAGE1', 'fundamental6', '2026-11-21T00:00:00Z')")
                conn.execute(
                    "INSERT INTO funnel_reference_db (sim_hash, expression, skeleton, settings_json, dataset_id, probe, stage, alpha_id, outcome, reason, sharpe, fitness, turnover, margin, updated_at) "
                    "VALUES ('h_real', 'rank(close)', 'rank', '{}', 'pv1', 'p1', 'STAGE1', 'A1', 'REJECTED', 'LOW_SHARPE', 0.42, 0.30, 0.20, 0.0005, '2026-11-21T00:00:00Z')"
                )
                conn.execute(
                    "INSERT INTO funnel_reference_db (sim_hash, expression, skeleton, settings_json, dataset_id, probe, stage, alpha_id, outcome, reason, sharpe, fitness, turnover, margin, updated_at) "
                    "VALUES ('h_fake1', 'rank(open)', 'rank', '{}', 'analyst4', 'p1', 'STAGE1', NULL, 'REJECTED', 'MISSING_METRICS', NULL, NULL, NULL, NULL, '2026-11-21T00:00:00Z')"
                )
                conn.commit()
            finally:
                conn.close()

            res = funnel.purge_false_failures()
            self.assertEqual(res["before_tried"], 3)
            self.assertEqual(res["after_tried"], 1)
            self.assertEqual(res["deleted_tried"], 2)
            self.assertEqual(res["deleted_ref"], 1)

    def test_headless_miner_has_circuit_breaker_and_sample_error_logging(self) -> None:
        src = Path("scripts/headless_brain_miner.py").read_text(encoding="utf-8")
        self.assertIn("CONSEC_FAIL_RECOVERY", src)
        self.assertIn("CIRCUIT_BREAKER", src)
        self.assertIn("sample_err=", src)


if __name__ == "__main__":
    unittest.main()
