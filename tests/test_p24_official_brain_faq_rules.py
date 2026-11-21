"""Contract tests for P24: Official WorldQuant BRAIN IS-Test FAQ Rules (Backfill, Weight/Coverage, Low-Turnover, SLOW/FAST)."""
from __future__ import annotations

import gc
import tempfile
import unittest
from pathlib import Path

from worldquant_brain.two_stage_funnel import STAGE2_NEUTRALIZATIONS, TwoStageAlphaFunnel


class TestP24OfficialBrainFaqRules(unittest.TestCase):
    def test_dataset_calibrated_backfill_days_avoid_over_backfilling(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            f = TwoStageAlphaFunnel(Path(td) / "vault.db")
            fnd = f.next_batch({"fundamental6": ["fnd6_ebitda"]}, batch_size=5)
            pv = f.next_batch({"pv1": ["close"]}, batch_size=5)
            nws = f.next_batch({"news18": ["nws18_sent"]}, batch_size=5)
            self.assertTrue(any("ts_backfill(fnd6_ebitda, 60)" in x["expression"] for x in fnd))
            self.assertTrue(any("ts_backfill(close, 2)" in x["expression"] for x in pv))
            self.assertTrue(any("ts_backfill(nws18_sent, 5)" in x["expression"] for x in nws))
            gc.collect()

    def test_weight_failure_uses_group_rank_and_trunc05(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            f = TwoStageAlphaFunnel(Path(td) / "vault.db")
            res = {
                "expression": "ts_zscore(ts_backfill(fnd6_ebitda, 60), 60)",
                "settings": dict(f.base_settings),
                "dataset_id": "fundamental6",
                "probe": "zscore60",
                "sharpe": 1.40,
                "fitness": 1.05,
                "turnover": 0.20,
                "margin": 0.0015,
                "checks_passed": False,
                "failed_checks": ["CONCENTRATED_WEIGHT"],
            }
            f._enqueue_immediate_refinement(res, "FAILED_CHECKS(CONCENTRATED_WEIGHT)")
            q = f.next_stage2_batch(batch_size=4)
            self.assertTrue(any("group_rank(" in x["expression"] and x["settings"]["truncation"] == 0.05 for x in q))
            gc.collect()

    def test_low_turnover_signal_boosts_returns_with_decay0_top1000_and_slow_fast_neuts(self) -> None:
        self.assertIn("SLOW", STAGE2_NEUTRALIZATIONS)
        self.assertIn("FAST", STAGE2_NEUTRALIZATIONS)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            f = TwoStageAlphaFunnel(Path(td) / "vault.db")
            res = {
                "expression": "rank(ts_zscore(ts_backfill(fnd6_ebitda, 60), 120))",
                "settings": dict(f.base_settings),
                "dataset_id": "fundamental6",
                "probe": "zscore60",
                "sharpe": 1.20,
                "fitness": 0.90,
                "turnover": 0.04,
                "margin": 0.0012,
                "checks_passed": False,
                "failed_checks": ["LOW_FITNESS"],
            }
            f._enqueue_immediate_refinement(res, "LOW_FITNESS")
            q = f.next_stage2_batch(batch_size=4)
            self.assertTrue(any(x["settings"].get("decay") == 0 and x["settings"].get("universe") == "TOP1000" for x in q))
            gc.collect()


if __name__ == "__main__":
    unittest.main()
