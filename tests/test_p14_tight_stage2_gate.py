"""Unit tests for P14 Tight Stage-2 Near-Miss Gate & Historical Tried Tracking."""
from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.headless_brain_miner import sync_vault_snapshot_to_drive
from worldquant_brain.two_stage_funnel import TwoStageAlphaFunnel, classify_verdict


class TestP14TightStage2Gate(unittest.TestCase):
    def test_rejects_weak_sharpe_fitness_turnover_margin_and_3_failed_checks(self) -> None:
        # Sharpe 1.10 (< 1.15) -> REJECTED
        v, _ = classify_verdict({"sharpe": 1.10, "fitness": 0.95, "turnover": 0.20, "margin": 0.0012, "checks_passed": False, "failed_checks": ["LOW_SHARPE"]})
        self.assertEqual(v, "REJECTED")

        # Fitness 0.80 (< 0.88) with Sharpe 1.18 -> REJECTED
        v, _ = classify_verdict({"sharpe": 1.18, "fitness": 0.80, "turnover": 0.20, "margin": 0.0012, "checks_passed": False, "failed_checks": ["LOW_FITNESS"]})
        self.assertEqual(v, "REJECTED")

        # Turnover 0.85 (> 0.78) -> REJECTED
        v, _ = classify_verdict({"sharpe": 1.20, "fitness": 0.95, "turnover": 0.85, "margin": 0.0012, "checks_passed": False, "failed_checks": ["HIGH_TURNOVER"]})
        self.assertEqual(v, "REJECTED")

        # Margin 4 bps (< 7 bps) with Sharpe 1.20 -> REJECTED
        v, _ = classify_verdict({"sharpe": 1.20, "fitness": 0.95, "turnover": 0.20, "margin": 0.0004, "checks_passed": False, "failed_checks": ["LOW_MARGIN"]})
        self.assertEqual(v, "REJECTED")

        # 3 failed checks (> 2) -> REJECTED
        v, _ = classify_verdict({"sharpe": 1.20, "fitness": 0.92, "turnover": 0.20, "margin": 0.0012, "checks_passed": False, "failed_checks": ["C1", "C2", "C3"]})
        self.assertEqual(v, "REJECTED")

    def test_accepts_genuinely_rescuable_near_miss_and_strong_negative_sharpe(self) -> None:
        # Sharpe 1.18, Fitness 0.90, Turnover 0.74, Margin 8 bps, 2 failed checks -> NEAR_MISS
        v, _ = classify_verdict({"sharpe": 1.18, "fitness": 0.90, "turnover": 0.74, "margin": 0.0008, "checks_passed": False, "failed_checks": ["LOW_SHARPE", "HIGH_TURNOVER"]})
        self.assertEqual(v, "NEAR_MISS")

        # Strong negative Sharpe (-1.45, Fitness -0.85, Margin -0.0006) -> NEAR_MISS for negation flip
        v, _ = classify_verdict({"sharpe": -1.45, "fitness": -0.85, "turnover": 0.22, "margin": -0.0006, "checks_passed": False, "failed_checks": ["LOW_SHARPE"]})
        self.assertEqual(v, "NEAR_MISS")

        # Delay 0 requires |Sharpe| >= 1.80 for NEAR_MISS
        v_low, _ = classify_verdict({"sharpe": 1.65, "fitness": 0.95, "turnover": 0.20, "margin": 0.0012, "checks_passed": False, "failed_checks": ["LOW_SHARPE"], "settings": {"delay": 0}})
        self.assertEqual(v_low, "REJECTED")
        v_ok, _ = classify_verdict({"sharpe": 1.85, "fitness": 0.95, "turnover": 0.20, "margin": 0.0012, "checks_passed": False, "failed_checks": ["LOW_SHARPE"], "settings": {"delay": 0}})
        self.assertEqual(v_ok, "NEAR_MISS")

    def test_funnel_defaults_and_historical_tried_total_in_status_json(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db = Path(td) / "vault.db"
            funnel = TwoStageAlphaFunnel(str(db))
            self.assertAlmostEqual(funnel.seed_sharpe_threshold, 1.15, places=4)
            self.assertEqual(funnel.max_stage2_variants_per_seed, 4)
            self.assertEqual(funnel.stage1_to_stage2_ratio, 3)

            conn = sqlite3.connect(str(db))
            try:
                conn.execute("INSERT INTO funnel_tried (sim_hash, expression, settings_json, stage, dataset_id, created_at) VALUES ('v1_0', 'v1_1', 'v1_2', 'v1_3', 'v1_4', 'v1_5')")
                conn.execute("INSERT INTO funnel_tried (sim_hash, expression, settings_json, stage, dataset_id, created_at) VALUES ('v2_0', 'v2_1', 'v2_2', 'v2_3', 'v2_4', 'v2_5')")
                conn.commit()
            finally:
                conn.close()

            status_json = Path(td) / "brain_pending_status.json"
            sync_vault_snapshot_to_drive(db, mirror_dir=Path(td) / "mirror", status_json_path=status_json)
            data = json.loads(status_json.read_text(encoding="utf-8"))
            self.assertEqual(data["historical_tried_total"], 2)


if __name__ == "__main__":
    unittest.main()
